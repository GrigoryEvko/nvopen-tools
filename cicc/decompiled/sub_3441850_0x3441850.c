// Function: sub_3441850
// Address: 0x3441850
//
__int64 __fastcall sub_3441850(__int64 *a1, __m128i a2)
{
  __int128 v4; // rax
  __int64 v5; // r9
  unsigned __int8 *v6; // r12
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 *v10; // rdx
  __int64 v11; // r14
  __int64 v12; // r15
  __int128 v13; // rax
  __int64 v14; // rax
  int v15; // edx
  int v16; // edi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int128 v19; // [rsp-A8h] [rbp-A8h]
  __int128 v20; // [rsp-98h] [rbp-98h]
  __int64 v21; // [rsp-78h] [rbp-78h]
  _QWORD *v22; // [rsp-70h] [rbp-70h]
  __int64 v23; // [rsp-68h] [rbp-68h]
  __int64 v24; // [rsp-60h] [rbp-60h]
  unsigned __int64 v25; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v26; // [rsp-40h] [rbp-40h]

  if ( *(_QWORD *)*a1 )
    return *(_QWORD *)*a1;
  v26 = *(_DWORD *)a1[1];
  if ( v26 > 0x40 )
  {
    sub_C43690((__int64)&v25, 0, 0);
    if ( v26 > 0x40 )
    {
      *(_QWORD *)v25 |= 0x8000000000000000LL;
      goto LABEL_6;
    }
  }
  else
  {
    v25 = 0;
  }
  v25 |= 0x8000000000000000LL;
LABEL_6:
  *(_QWORD *)&v4 = sub_34007B0(a1[2], (__int64)&v25, a1[3], *(_DWORD *)a1[4], *(_QWORD *)(a1[4] + 8), 0, a2, 0);
  v6 = sub_3406EB0(
         (_QWORD *)a1[2],
         0xBAu,
         a1[3],
         *(unsigned int *)a1[4],
         *(_QWORD *)(a1[4] + 8),
         v5,
         *(_OWORD *)a1[5],
         v4);
  v7 = (__int64 *)a1[6];
  v9 = v8;
  v10 = (__int64 *)a1[7];
  v21 = a1[3];
  v22 = (_QWORD *)a1[2];
  v11 = *v10;
  v12 = v10[1];
  v24 = *v7;
  v23 = v7[1];
  *(_QWORD *)&v13 = sub_33ED040(v22, 0x16u);
  *((_QWORD *)&v20 + 1) = v12;
  *(_QWORD *)&v20 = v11;
  *((_QWORD *)&v19 + 1) = v9;
  *(_QWORD *)&v19 = v6;
  v14 = sub_340F900(v22, 0xD0u, v21, v24, v23, v21, v19, v20, v13);
  v16 = v15;
  v17 = v14;
  v18 = *a1;
  *(_QWORD *)v18 = v17;
  *(_DWORD *)(v18 + 8) = v16;
  if ( v26 > 0x40 )
  {
    if ( v25 )
      j_j___libc_free_0_0(v25);
  }
  return *(_QWORD *)*a1;
}
