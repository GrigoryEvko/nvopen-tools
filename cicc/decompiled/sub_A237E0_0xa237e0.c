// Function: sub_A237E0
// Address: 0xa237e0
//
__int64 __fastcall sub_A237E0(__int64 *a1)
{
  __int64 v2; // r13
  __int64 v3; // rax
  _QWORD *v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdi
  volatile signed __int32 *v17; // rax
  unsigned int v18; // r12d
  __int64 v20; // [rsp+0h] [rbp-40h] BYREF
  volatile signed __int32 *v21; // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+10h] [rbp-30h] BYREF
  volatile signed __int32 *v23; // [rsp+18h] [rbp-28h]

  sub_A23770(&v20);
  sub_A186C0(v20, 7, 1);
  v2 = v20;
  v3 = *(unsigned int *)(v20 + 8);
  if ( v3 + 1 > (unsigned __int64)*(unsigned int *)(v20 + 12) )
  {
    sub_C8D5F0(v20, v20 + 16, v3 + 1, 16);
    v3 = *(unsigned int *)(v2 + 8);
  }
  v4 = (_QWORD *)(*(_QWORD *)v2 + 16 * v3);
  *v4 = 1;
  v4[1] = 2;
  ++*(_DWORD *)(v2 + 8);
  v5 = v20;
  v6 = *(unsigned int *)(v20 + 8);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(v20 + 12) )
  {
    sub_C8D5F0(v20, v20 + 16, v6 + 1, 16);
    v6 = *(unsigned int *)(v5 + 8);
  }
  v7 = (_QWORD *)(*(_QWORD *)v5 + 16 * v6);
  *v7 = 6;
  v7[1] = 4;
  ++*(_DWORD *)(v5 + 8);
  v8 = v20;
  v9 = *(unsigned int *)(v20 + 8);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v20 + 12) )
  {
    sub_C8D5F0(v20, v20 + 16, v9 + 1, 16);
    v9 = *(unsigned int *)(v8 + 8);
  }
  v10 = (_QWORD *)(*(_QWORD *)v8 + 16 * v9);
  *v10 = 8;
  v10[1] = 4;
  ++*(_DWORD *)(v8 + 8);
  v11 = v20;
  v12 = *(unsigned int *)(v20 + 8);
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v20 + 12) )
  {
    sub_C8D5F0(v20, v20 + 16, v12 + 1, 16);
    v12 = *(unsigned int *)(v11 + 8);
  }
  v13 = (_QWORD *)(*(_QWORD *)v11 + 16 * v12);
  *v13 = 6;
  v14 = v20;
  v13[1] = 4;
  ++*(_DWORD *)(v11 + 8);
  sub_A186C0(v14, 6, 4);
  sub_A186C0(v20, 1, 2);
  v15 = v20;
  v16 = *a1;
  v20 = 0;
  v22 = v15;
  v17 = v21;
  v21 = 0;
  v23 = v17;
  v18 = sub_A1AB30(v16, &v22);
  if ( v23 )
    sub_A191D0(v23);
  if ( v21 )
    sub_A191D0(v21);
  return v18;
}
