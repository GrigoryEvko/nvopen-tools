// Function: sub_2BDE770
// Address: 0x2bde770
//
void __fastcall sub_2BDE770(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __m128i *v5; // rax
  __int64 v6; // rcx
  void (__fastcall *v7)(size_t *, __int64, __int64); // rax
  unsigned __int64 v8; // r9
  size_t v9; // r13
  __m128i *v10; // rbx
  __m128i *v11; // r10
  unsigned __int64 v12; // rdx
  int v13; // ecx
  unsigned __int64 v14; // r12
  __m128i *v15; // r15
  int v16; // r14d
  __int8 v17; // cl
  unsigned __int64 v18; // rcx
  __int64 v19; // r8
  unsigned __int64 v20; // r8
  __int8 v21; // cl
  __int8 v22; // si
  size_t v23; // r11
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // r13
  void *v27; // rax
  void *v28; // rax
  void *v29; // rax
  void *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // [rsp-A4h] [rbp-A4h]
  unsigned __int64 v34; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v35; // [rsp-A0h] [rbp-A0h]
  __m128i *v36; // [rsp-98h] [rbp-98h]
  __m128i *v37; // [rsp-98h] [rbp-98h]
  __int8 *v38; // [rsp-90h] [rbp-90h]
  __m128i *v39; // [rsp-88h] [rbp-88h]
  unsigned __int64 v40; // [rsp-80h] [rbp-80h]
  __m128i v41; // [rsp-78h] [rbp-78h] BYREF
  __int64 *v42; // [rsp-68h] [rbp-68h] BYREF
  size_t v43; // [rsp-60h] [rbp-60h] BYREF
  __int64 v44; // [rsp-58h] [rbp-58h] BYREF
  void (__fastcall *v45)(size_t *, size_t *, __int64); // [rsp-50h] [rbp-50h]
  __int64 v46; // [rsp-48h] [rbp-48h]

  if ( a3 )
  {
    v42 = &v44;
    sub_2BDC240((__int64 *)&v42, a2, (__int64)&a2[a3]);
    v5 = (__m128i *)sub_2240FD0((unsigned __int64 *)&v42, v43, 0, 1u, 0);
    v39 = &v41;
    if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
    {
      v41 = _mm_loadu_si128(v5 + 1);
    }
    else
    {
      v39 = (__m128i *)v5->m128i_i64[0];
      v41.m128i_i64[0] = v5[1].m128i_i64[0];
    }
    v6 = v5->m128i_i64[1];
    v5[1].m128i_i8[0] = 0;
    v40 = v6;
    v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
    v5->m128i_i64[1] = 0;
    if ( v42 != &v44 )
      j_j___libc_free_0((unsigned __int64)v42);
    v7 = *(void (__fastcall **)(size_t *, __int64, __int64))(a4 + 16);
    v42 = (__int64 *)a1;
    v45 = 0;
    v8 = v40;
    if ( v7 )
    {
      v7(&v43, a4, 2);
      v8 = v40;
      v46 = *(_QWORD *)(a4 + 24);
      v45 = *(void (__fastcall **)(size_t *, size_t *, __int64))(a4 + 16);
    }
    v9 = 0;
    v38 = 0;
    if ( &v39->m128i_i8[v8] != (__int8 *)v39 )
    {
      v10 = v39;
      v11 = v39;
      LODWORD(v12) = 0;
      v13 = 0;
      v14 = 0;
      v15 = (__m128i *)((char *)v39 + v8);
      v16 = 0;
      do
      {
LABEL_14:
        if ( v13 != 1 )
          goto LABEL_32;
        while ( 1 )
        {
          v17 = v10->m128i_i8[0];
          if ( v10->m128i_i8[0] == 60 )
            break;
          if ( v17 != 62 )
          {
            if ( !v17 )
            {
              v30 = sub_CB72A0();
              v31 = sub_904010(
                      (__int64)v30,
                      "Missing '>' in pass pipeline. End-of-string reached while reading arguments for pass '");
              v32 = sub_A51340(v31, v38, v9);
              sub_904010(v32, "'.\n");
              exit(1);
            }
LABEL_12:
            v13 = 1;
LABEL_13:
            v10 = (__m128i *)((char *)v10 + 1);
            ++v14;
            if ( v15 == v10 )
              goto LABEL_17;
            goto LABEL_14;
          }
          if ( --v16 )
          {
            if ( v16 < 0 )
            {
              v27 = sub_CB72A0();
              sub_904010((__int64)v27, "Unexpected '>' in pass pipeline.\n");
              exit(1);
            }
            goto LABEL_12;
          }
          v18 = v33;
          if ( v33 > v8 )
            v18 = v8;
          v19 = 0;
          if ( v14 >= v18 )
          {
            v20 = v14;
            if ( v8 <= v14 )
              v20 = v8;
            v19 = v20 - v18;
          }
          v34 = v8;
          v36 = v11;
          sub_2BDE5C0((__int64 *)&v42, v38, v9, (__int64)v11->m128i_i64 + v18, v19);
          v11 = v36;
          v8 = v34;
          if ( v15 == (__m128i *)&v10->m128i_i8[1] )
            goto LABEL_17;
          v21 = v10->m128i_i8[1];
          if ( v21 && v21 != 44 )
          {
            v28 = sub_CB72A0();
            sub_904010((__int64)v28, "Expected delimiter or end-of-string after pass arguments.\n");
            exit(1);
          }
          v10 = (__m128i *)((char *)v10 + 2);
          LODWORD(v12) = v14 + 2;
          v14 += 2LL;
          if ( v10 == v15 )
            goto LABEL_17;
LABEL_32:
          v22 = v10->m128i_i8[0];
          if ( v10->m128i_i8[0] != 60 )
          {
            if ( v22 == 62 )
            {
              v29 = sub_CB72A0();
              sub_904010((__int64)v29, "Unexpected '>' in pass pipeline.\n");
              exit(1);
            }
            if ( !v22 || (v13 = 0, v22 == 44) )
            {
              v12 = (int)v12;
              if ( (int)v12 > v8 )
                v12 = v8;
              v23 = 0;
              if ( v14 >= v12 )
              {
                v24 = v14;
                if ( v8 <= v14 )
                  v24 = v8;
                v23 = v24 - v12;
              }
              v35 = v8;
              v37 = v11;
              sub_2BDE5C0((__int64 *)&v42, &v11->m128i_i8[v12], v23, 0, 0);
              v11 = v37;
              v8 = v35;
              v13 = 0;
              LODWORD(v12) = v14 + 1;
            }
            goto LABEL_13;
          }
          v25 = (int)v12;
          if ( (int)v12 > v8 )
            v25 = v8;
          v26 = v25;
          if ( v25 <= v14 )
          {
            v26 = v14;
            if ( v8 <= v14 )
              v26 = v8;
          }
          v10 = (__m128i *)((char *)v10 + 1);
          v9 = v26 - v25;
          v38 = &v11->m128i_i8[v25];
          ++v16;
          v33 = ++v14;
          if ( v15 == v10 )
            goto LABEL_17;
        }
        v10 = (__m128i *)((char *)v10 + 1);
        ++v16;
        v13 = 1;
        ++v14;
      }
      while ( v15 != v10 );
    }
LABEL_17:
    if ( v45 )
      v45(&v43, &v43, 3);
    if ( v39 != &v41 )
      j_j___libc_free_0((unsigned __int64)v39);
  }
}
