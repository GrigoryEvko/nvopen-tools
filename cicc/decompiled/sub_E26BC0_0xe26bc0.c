// Function: sub_E26BC0
// Address: 0xe26bc0
//
unsigned __int64 __fastcall sub_E26BC0(__int64 a1, __int64 **a2, unsigned __int64 *a3)
{
  __int64 v4; // rdx
  unsigned __int64 v5; // r12
  __int64 v6; // rdx
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rax
  _BYTE *v9; // rdx
  unsigned __int64 *v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // rax
  __int64 *v14; // r13
  __int64 v15; // rax
  __int64 *v16; // rax
  unsigned __int64 *v17; // [rsp+8h] [rbp-38h]

  v4 = **a2;
  v5 = (v4 + (*a2)[1] + 7) & 0xFFFFFFFFFFFFFFF8LL;
  (*a2)[1] = v5 - v4 + 40;
  if ( (*a2)[1] > (unsigned __int64)(*a2)[2] )
  {
    v13 = (__int64 *)sub_22077B0(32);
    v14 = v13;
    if ( v13 )
    {
      *v13 = 0;
      v13[1] = 0;
      v13[2] = 0;
      v13[3] = 0;
    }
    v15 = sub_2207820(4096);
    v14[2] = 4096;
    *v14 = v15;
    v5 = v15;
    v16 = *a2;
    v14[1] = 40;
    v14[3] = (__int64)v16;
    *a2 = v14;
  }
  if ( !v5 )
  {
    sub_E21AA0(a1, a3);
    MEMORY[0x18] = 0;
    BUG();
  }
  *(_DWORD *)(v5 + 8) = 24;
  *(_QWORD *)(v5 + 16) = 0;
  *(_QWORD *)(v5 + 24) = 0;
  *(_QWORD *)v5 = &unk_49E1268;
  *(_QWORD *)(v5 + 32) = 0;
  *(_DWORD *)(v5 + 24) = sub_E21AA0(a1, a3);
  *(_DWORD *)(v5 + 28) = sub_E21AC0(a1, a3);
  *(_DWORD *)(v5 + 32) = sub_E21AA0(a1, a3);
  *(_DWORD *)(v5 + 36) = sub_E21AA0(a1, a3);
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  v6 = **a2;
  v7 = (v6 + (*a2)[1] + 7) & 0xFFFFFFFFFFFFFFF8LL;
  (*a2)[1] = v7 - v6 + 40;
  if ( (*a2)[1] > (unsigned __int64)(*a2)[2] )
  {
    v11 = (unsigned __int64 *)sub_22077B0(32);
    if ( v11 )
    {
      *v11 = 0;
      v11[1] = 0;
      v11[2] = 0;
      v11[3] = 0;
    }
    v17 = v11;
    v7 = sub_2207820(4096);
    *v17 = v7;
    v12 = *a2;
    v17[2] = 4096;
    v17[3] = (unsigned __int64)v12;
    *a2 = (__int64 *)v17;
    v17[1] = 40;
  }
  if ( !v7 )
  {
    MEMORY[0x10] = sub_E263F0(a1, (__int64 *)a3, v5);
    BUG();
  }
  *(_BYTE *)(v7 + 24) = 0;
  *(_DWORD *)(v7 + 8) = 27;
  *(_QWORD *)(v7 + 16) = 0;
  *(_QWORD *)(v7 + 32) = 0;
  *(_QWORD *)v7 = &unk_49E11E0;
  *(_QWORD *)(v7 + 16) = sub_E263F0(a1, (__int64 *)a3, v5);
  v8 = *a3;
  v9 = (_BYTE *)a3[1];
  if ( *a3 && *v9 == 56 )
  {
    a3[1] = (unsigned __int64)(v9 + 1);
    *a3 = v8 - 1;
  }
  return v7;
}
