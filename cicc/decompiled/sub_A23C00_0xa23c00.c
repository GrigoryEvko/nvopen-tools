// Function: sub_A23C00
// Address: 0xa23c00
//
__int64 __fastcall sub_A23C00(__int64 *a1)
{
  __int64 v2; // r13
  __int64 v3; // rax
  _QWORD *v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdi
  volatile signed __int32 *v18; // rax
  unsigned int v19; // r12d
  __int64 v21; // [rsp+0h] [rbp-40h] BYREF
  volatile signed __int32 *v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+10h] [rbp-30h] BYREF
  volatile signed __int32 *v24; // [rsp+18h] [rbp-28h]

  sub_A23770(&v21);
  sub_A186C0(v21, 12, 1);
  v2 = v21;
  v3 = *(unsigned int *)(v21 + 8);
  if ( v3 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
  {
    sub_C8D5F0(v21, v21 + 16, v3 + 1, 16);
    v3 = *(unsigned int *)(v2 + 8);
  }
  v4 = (_QWORD *)(*(_QWORD *)v2 + 16 * v3);
  *v4 = 1;
  v4[1] = 2;
  ++*(_DWORD *)(v2 + 8);
  v5 = v21;
  v6 = *(unsigned int *)(v21 + 8);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
  {
    sub_C8D5F0(v21, v21 + 16, v6 + 1, 16);
    v6 = *(unsigned int *)(v5 + 8);
  }
  v7 = (_QWORD *)(*(_QWORD *)v5 + 16 * v6);
  *v7 = 6;
  v8 = v21;
  v7[1] = 4;
  ++*(_DWORD *)(v5 + 8);
  sub_A186C0(v8, 1, 2);
  v9 = v21;
  v10 = *(unsigned int *)(v21 + 8);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
  {
    sub_C8D5F0(v21, v21 + 16, v10 + 1, 16);
    v10 = *(unsigned int *)(v9 + 8);
  }
  v11 = (_QWORD *)(*(_QWORD *)v9 + 16 * v10);
  *v11 = 6;
  v11[1] = 4;
  ++*(_DWORD *)(v9 + 8);
  v12 = v21;
  v13 = *(unsigned int *)(v21 + 8);
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
  {
    sub_C8D5F0(v21, v21 + 16, v13 + 1, 16);
    v13 = *(unsigned int *)(v12 + 8);
  }
  v14 = (_QWORD *)(*(_QWORD *)v12 + 16 * v13);
  *v14 = 0;
  v15 = v21;
  v14[1] = 6;
  ++*(_DWORD *)(v12 + 8);
  sub_A186C0(v15, 6, 4);
  v16 = v21;
  v17 = *a1;
  v21 = 0;
  v23 = v16;
  v18 = v22;
  v22 = 0;
  v24 = v18;
  v19 = sub_A1AB30(v17, &v23);
  if ( v24 )
    sub_A191D0(v24);
  if ( v22 )
    sub_A191D0(v22);
  return v19;
}
