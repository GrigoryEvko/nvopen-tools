// Function: sub_334C7C0
// Address: 0x334c7c0
//
void __fastcall sub_334C7C0(__int64 *a1)
{
  __int64 *v1; // rdx
  __int64 v2; // r8
  __int64 v3; // r9
  __m128i v4; // xmm0
  __int64 v5; // rax
  __int64 *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rbx
  __int64 v11; // r15
  __int64 *v12; // rax
  bool v13; // zf
  char v14; // [rsp+1Fh] [rbp-141h]
  __m128i v15; // [rsp+20h] [rbp-140h] BYREF
  __int64 *v16; // [rsp+30h] [rbp-130h]
  __int64 *v17; // [rsp+38h] [rbp-128h]
  __int64 v18; // [rsp+40h] [rbp-120h]
  void *v19; // [rsp+50h] [rbp-110h] BYREF
  __int64 v20; // [rsp+58h] [rbp-108h]
  __int64 *v21; // [rsp+60h] [rbp-100h]
  __m128i v22; // [rsp+68h] [rbp-F8h] BYREF
  __int64 (__fastcall *v23)(__m128i *, __m128i *, int); // [rsp+78h] [rbp-E8h]
  __int64 *(__fastcall *v24)(__int64 *, __int64 *); // [rsp+80h] [rbp-E0h]
  __int64 v25; // [rsp+90h] [rbp-D0h] BYREF
  __int64 *v26; // [rsp+98h] [rbp-C8h]
  __int64 v27; // [rsp+A0h] [rbp-C0h]
  int v28; // [rsp+A8h] [rbp-B8h]
  char v29; // [rsp+ACh] [rbp-B4h]
  char v30; // [rsp+B0h] [rbp-B0h] BYREF

  sub_33E2990();
  v15.m128i_i64[0] = (__int64)&v25;
  v4 = _mm_loadu_si128(&v15);
  v26 = (__int64 *)&v30;
  v5 = a1[96];
  v29 = 1;
  v6 = (__int64 *)a1[51];
  v20 = v5;
  a1[96] = (__int64)&v19;
  v25 = 0;
  v21 = a1;
  v19 = &unk_4A366C8;
  v23 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_33255B0;
  v24 = sub_33259A0;
  v7 = *a1;
  v27 = 16;
  v15.m128i_i64[0] = v7;
  v8 = a1[2];
  v28 = 0;
  v15.m128i_i64[1] = v8;
  v16 = a1;
  v17 = &v25;
  v18 = 0;
  v22 = v4;
  do
  {
    if ( v6 == a1 + 50 )
      break;
    v14 = 0;
    v6 = a1 + 50;
    do
    {
      while ( 1 )
      {
        v9 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
        v10 = v9;
        v6 = (__int64 *)v9;
        if ( !v9 )
          BUG();
        v11 = v9 - 8;
        if ( !*(_QWORD *)(v9 + 48) && v11 != a1[48] )
        {
          v6 = *(__int64 **)(v9 + 8);
          sub_33EBEB0(a1, v9 - 8);
          goto LABEL_19;
        }
        if ( v29 )
        {
          v12 = v26;
          v1 = &v26[HIDWORD(v27)];
          if ( v26 != v1 )
          {
            while ( v11 != *v12 )
            {
              if ( v1 == ++v12 )
                goto LABEL_25;
            }
            goto LABEL_11;
          }
LABEL_25:
          if ( HIDWORD(v27) < (unsigned int)v27 )
          {
            ++HIDWORD(v27);
            *v1 = v11;
            ++v25;
            break;
          }
        }
        sub_C8CC70((__int64)&v25, v9 - 8, (__int64)v1, v9, v2, v3);
        if ( (_BYTE)v1 )
          break;
LABEL_11:
        if ( (__int64 *)a1[51] == v6 )
          goto LABEL_12;
      }
      sub_3349730(v15.m128i_i64, v11, (__int64)v1, v9, v2);
      v14 = 1;
      if ( *(_QWORD *)(v10 + 48) || v11 == a1[48] )
        goto LABEL_11;
      v6 = *(__int64 **)(v10 + 8);
      sub_33EBEB0(a1, v11);
LABEL_19:
      ;
    }
    while ( (__int64 *)a1[51] != v6 );
LABEL_12:
    ;
  }
  while ( v14 );
  sub_33F7860(a1);
  v19 = &unk_4A366C8;
  if ( v23 )
    v23(&v22, &v22, 3);
  v13 = v29 == 0;
  v21[96] = v20;
  if ( v13 )
    _libc_free((unsigned __int64)v26);
}
