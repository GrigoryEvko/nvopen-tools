// Function: sub_E279E0
// Address: 0xe279e0
//
unsigned __int64 __fastcall sub_E279E0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  _BYTE *v3; // rdx
  unsigned __int64 v4; // r14
  _QWORD *v5; // rax
  unsigned __int64 v6; // r12
  _QWORD *v7; // rax
  unsigned __int64 v8; // rsi
  __int64 *v10; // rax
  __int64 *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 *v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax

  v2 = *a2;
  if ( *a2 )
  {
    v3 = (_BYTE *)a2[1];
    if ( *v3 == 46 )
    {
      a2[1] = (__int64)(v3 + 1);
      *a2 = v2 - 1;
    }
  }
  v4 = sub_E27700(a1, a2, 2);
  if ( *(_BYTE *)(a1 + 8) || *a2 )
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  else
  {
    v5 = *(_QWORD **)(a1 + 16);
    v6 = (*v5 + v5[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    v5[1] = v6 - *v5 + 40;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v10 = (__int64 *)sub_22077B0(32);
      v11 = v10;
      if ( v10 )
      {
        *v10 = 0;
        v10[1] = 0;
        v10[2] = 0;
        v10[3] = 0;
      }
      v12 = sub_2207820(4096);
      v11[2] = 4096;
      *v11 = v12;
      v6 = v12;
      v13 = *(_QWORD *)(a1 + 16);
      v11[1] = 40;
      v11[3] = v13;
      *(_QWORD *)(a1 + 16) = v11;
    }
    if ( !v6 )
    {
      MEMORY[0x20] = v4;
      BUG();
    }
    *(_BYTE *)(v6 + 24) = 0;
    *(_QWORD *)(v6 + 32) = 0;
    *(_QWORD *)(v6 + 32) = v4;
    *(_DWORD *)(v6 + 8) = 27;
    *(_QWORD *)(v6 + 16) = 0;
    *(_QWORD *)v6 = &unk_49E11E0;
    v7 = *(_QWORD **)(a1 + 16);
    v8 = (*v7 + v7[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    v7[1] = v8 - *v7 + 40;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v14 = (__int64 *)sub_22077B0(32);
      v15 = v14;
      if ( v14 )
      {
        *v14 = 0;
        v14[1] = 0;
        v14[2] = 0;
        v14[3] = 0;
      }
      v16 = sub_2207820(4096);
      v15[2] = 4096;
      *v15 = v16;
      v8 = v16;
      v17 = *(_QWORD *)(a1 + 16);
      v15[1] = 40;
      v15[3] = v17;
      *(_QWORD *)(a1 + 16) = v15;
    }
    if ( !v8 )
    {
      MEMORY[0x18] = 0;
      BUG();
    }
    *(_QWORD *)(v8 + 24) = 0;
    *(_QWORD *)(v8 + 32) = 0;
    *(_DWORD *)(v8 + 8) = 5;
    *(_QWORD *)v8 = &unk_49E0F88;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)(v8 + 24) = 27;
    *(_QWORD *)(v8 + 32) = "`RTTI Type Descriptor Name'";
    *(_QWORD *)(v6 + 16) = sub_E20AE0((__int64 **)(a1 + 16), v8);
    return v6;
  }
}
