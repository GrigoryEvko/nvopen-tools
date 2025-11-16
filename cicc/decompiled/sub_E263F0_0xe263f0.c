// Function: sub_E263F0
// Address: 0xe263f0
//
unsigned __int64 __fastcall sub_E263F0(__int64 a1, __int64 *a2, __int64 a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // r13
  __int64 v6; // r14
  __int64 v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 **v13; // r8
  unsigned __int64 v14; // r12
  __int64 *v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD **)(a1 + 16);
  v5 = (_QWORD *)((*v4 + v4[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
  v4[1] = (char *)v5 - *v4 + 16;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v16 = (__int64 *)sub_22077B0(32);
    v17 = v16;
    if ( v16 )
    {
      *v16 = 0;
      v16[1] = 0;
      v16[2] = 0;
      v16[3] = 0;
    }
    v18 = sub_2207820(4096);
    v17[2] = 4096;
    *v17 = v18;
    v5 = (_QWORD *)v18;
    v19 = *(_QWORD *)(a1 + 16);
    v17[1] = 16;
    v17[3] = v19;
    *(_QWORD *)(a1 + 16) = v17;
  }
  if ( !v5 )
  {
    MEMORY[0] = a3;
    BUG();
  }
  *v5 = 0;
  v5[1] = 0;
  *v5 = a3;
  v6 = 1;
  while ( !(unsigned __int8)sub_E20730((size_t *)a2, 1u, "@") )
  {
    v8 = *(_QWORD **)(a1 + 16);
    ++v6;
    v9 = (_QWORD *)((*v8 + v8[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v8[1] = (char *)v9 + 16LL - *v8;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v10 = (_QWORD *)sub_22077B0(32);
      if ( v10 )
      {
        *v10 = 0;
        v10[1] = 0;
        v10[2] = 0;
        v10[3] = 0;
      }
      v24 = v10;
      v9 = (_QWORD *)sub_2207820(4096);
      *v24 = v9;
      v11 = *(_QWORD *)(a1 + 16);
      v24[2] = 4096;
      v24[3] = v11;
      *(_QWORD *)(a1 + 16) = v24;
      v24[1] = 16;
    }
    if ( !v9 )
    {
      MEMORY[8] = v5;
      BUG();
    }
    v9[1] = 0;
    *v9 = 0;
    v9[1] = v5;
    if ( !*a2 )
    {
      *(_BYTE *)(a1 + 8) = 1;
      return 0;
    }
    v7 = sub_E26270(a1, a2);
    if ( *(_BYTE *)(a1 + 8) )
      return 0;
    *v9 = v7;
    v5 = v9;
  }
  v12 = *(_QWORD **)(a1 + 16);
  v13 = (__int64 **)(a1 + 16);
  v14 = (*v12 + v12[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v12[1] = v14 - *v12 + 24;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v20 = (__int64 *)sub_22077B0(32);
    v21 = v20;
    if ( v20 )
    {
      *v20 = 0;
      v20[1] = 0;
      v20[2] = 0;
      v20[3] = 0;
    }
    v22 = sub_2207820(4096);
    v21[2] = 4096;
    v13 = (__int64 **)(a1 + 16);
    v14 = v22;
    *v21 = v22;
    v23 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = v21;
    v21[3] = v23;
    v21[1] = 24;
    if ( v14 )
      goto LABEL_15;
LABEL_27:
    MEMORY[0x10] = sub_E208B0(v13, v5, v6);
    BUG();
  }
  if ( !v14 )
    goto LABEL_27;
LABEL_15:
  *(_DWORD *)(v14 + 8) = 20;
  *(_QWORD *)(v14 + 16) = 0;
  *(_QWORD *)v14 = &unk_49E1240;
  *(_QWORD *)(v14 + 16) = sub_E208B0(v13, v5, v6);
  return v14;
}
