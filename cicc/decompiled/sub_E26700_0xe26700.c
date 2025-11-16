// Function: sub_E26700
// Address: 0xe26700
//
unsigned __int64 __fastcall sub_E26700(__int64 a1, __int64 *a2, char a3)
{
  _QWORD *v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r15
  _QWORD *v7; // rax
  unsigned __int64 v8; // r12
  unsigned __int64 *v10; // rax
  unsigned __int64 v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 *v16; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD **)(a1 + 16);
  v5 = (*v4 + v4[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v4[1] = v5 - *v4 + 32;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v12 = (__int64 *)sub_22077B0(32);
    v13 = v12;
    if ( v12 )
    {
      *v12 = 0;
      v12[1] = 0;
      v12[2] = 0;
      v12[3] = 0;
    }
    v14 = sub_2207820(4096);
    v13[2] = 4096;
    *v13 = v14;
    v5 = v14;
    v15 = *(_QWORD *)(a1 + 16);
    v13[1] = 32;
    v13[3] = v15;
    *(_QWORD *)(a1 + 16) = v13;
  }
  if ( !v5 )
  {
    MEMORY[0x18] = 0;
    BUG();
  }
  *(_BYTE *)(v5 + 24) = 0;
  *(_BYTE *)(v5 + 24) = a3;
  *(_DWORD *)(v5 + 8) = 7;
  *(_QWORD *)(v5 + 16) = 0;
  *(_QWORD *)v5 = &unk_49E0FD8;
  *(_DWORD *)(v5 + 28) = 0;
  v6 = sub_E263F0(a1, a2, v5);
  v7 = *(_QWORD **)(a1 + 16);
  v8 = (*v7 + v7[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v7[1] = v8 - *v7 + 32;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v10 = (unsigned __int64 *)sub_22077B0(32);
    if ( v10 )
    {
      *v10 = 0;
      v10[1] = 0;
      v10[2] = 0;
      v10[3] = 0;
    }
    v16 = v10;
    v8 = sub_2207820(4096);
    *v16 = v8;
    v11 = *(_QWORD *)(a1 + 16);
    v16[2] = 4096;
    v16[3] = v11;
    *(_QWORD *)(a1 + 16) = v16;
    v16[1] = 32;
  }
  if ( !v8 )
  {
    MEMORY[0x10] = v6;
    BUG();
  }
  *(_BYTE *)(v8 + 24) = 0;
  *(_QWORD *)(v8 + 16) = 0;
  *(_QWORD *)(v8 + 16) = v6;
  *(_DWORD *)(v8 + 8) = 25;
  *(_QWORD *)v8 = &unk_49E1290;
  if ( (unsigned __int8)sub_E20730((size_t *)a2, 3u, "4IA") )
  {
    *(_BYTE *)(v8 + 24) = 0;
    if ( !*a2 )
      return v8;
LABEL_10:
    *(_DWORD *)(v5 + 28) = sub_E21AA0(a1, (unsigned __int64 *)a2);
    return v8;
  }
  if ( !(unsigned __int8)sub_E20730((size_t *)a2, 1u, "5") )
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  *(_BYTE *)(v8 + 24) = 1;
  if ( *a2 )
    goto LABEL_10;
  return v8;
}
