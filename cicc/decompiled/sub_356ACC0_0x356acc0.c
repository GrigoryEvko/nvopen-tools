// Function: sub_356ACC0
// Address: 0x356acc0
//
_QWORD *__fastcall sub_356ACC0(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r9
  __int64 *v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  __m128i v17; // [rsp+10h] [rbp-150h] BYREF
  char v18; // [rsp+20h] [rbp-140h]
  __int64 v19; // [rsp+30h] [rbp-130h] BYREF
  __int64 *v20; // [rsp+38h] [rbp-128h]
  __int64 v21; // [rsp+40h] [rbp-120h]
  int v22; // [rsp+48h] [rbp-118h]
  char v23; // [rsp+4Ch] [rbp-114h]
  _QWORD v24[8]; // [rsp+50h] [rbp-110h] BYREF
  unsigned __int64 v25; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v26; // [rsp+98h] [rbp-C8h]
  __int64 v27; // [rsp+A0h] [rbp-C0h]
  unsigned __int64 v28[22]; // [rsp+B0h] [rbp-B0h] BYREF

  v3 = a2[4];
  v20 = v24;
  memset(v28, 0, 0x78u);
  v4 = *a2;
  v28[1] = (unsigned __int64)&v28[4];
  LODWORD(v28[2]) = 8;
  BYTE4(v28[3]) = 1;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v21 = 0x100000008LL;
  v22 = 0;
  v23 = 1;
  v24[0] = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v19 = 1;
  v17.m128i_i64[0] = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v18 = 0;
  sub_356AC80(&v25, &v17);
  if ( !v23 )
    goto LABEL_8;
  v8 = v20;
  v6 = HIDWORD(v21);
  v5 = &v20[HIDWORD(v21)];
  if ( v20 == v5 )
  {
LABEL_7:
    if ( HIDWORD(v21) >= (unsigned int)v21 )
    {
LABEL_8:
      sub_C8CC70((__int64)&v19, v3, (__int64)v5, v6, (__int64)&v19, v7);
      goto LABEL_9;
    }
    ++HIDWORD(v21);
    *v5 = v3;
    ++v19;
  }
  else
  {
    while ( v3 != *v8 )
    {
      if ( v5 == ++v8 )
        goto LABEL_7;
    }
  }
LABEL_9:
  sub_C8CF70((__int64)a1, a1 + 4, 8, (__int64)v24, (__int64)&v19);
  v9 = v25;
  v25 = 0;
  a1[12] = v9;
  v10 = v26;
  v26 = 0;
  a1[13] = v10;
  v11 = v27;
  v27 = 0;
  a1[14] = v11;
  sub_C8CF70((__int64)(a1 + 15), a1 + 19, 8, (__int64)&v28[4], (__int64)v28);
  v12 = v28[12];
  v13 = v25;
  v28[12] = 0;
  a1[27] = v12;
  v14 = v28[13];
  v28[13] = 0;
  a1[28] = v14;
  v15 = v28[14];
  v28[14] = 0;
  a1[29] = v15;
  if ( v13 )
    j_j___libc_free_0(v13);
  if ( !v23 )
    _libc_free((unsigned __int64)v20);
  if ( v28[12] )
    j_j___libc_free_0(v28[12]);
  if ( !BYTE4(v28[3]) )
    _libc_free(v28[1]);
  return a1;
}
