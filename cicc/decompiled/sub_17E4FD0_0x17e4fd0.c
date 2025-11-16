// Function: sub_17E4FD0
// Address: 0x17e4fd0
//
__int64 *__fastcall sub_17E4FD0(__int64 *a1, _QWORD *a2, __int64 a3)
{
  _DWORD *v5; // r13
  int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v10; // rdx
  char *v11; // rsi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rcx
  __m128i *v15; // rax
  __m128i *v16; // rcx
  __m128i *v17; // rdx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rcx
  __m128i *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdi
  __int64 v24; // rax
  __m128i *v25; // rax
  __int64 v26; // rcx
  __m128i *v27; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v28; // [rsp+38h] [rbp-C8h]
  __m128i v29; // [rsp+40h] [rbp-C0h] BYREF
  _QWORD *v30; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v31; // [rsp+58h] [rbp-A8h]
  _QWORD v32[2]; // [rsp+60h] [rbp-A0h] BYREF
  _QWORD *v33; // [rsp+70h] [rbp-90h] BYREF
  __int64 v34; // [rsp+78h] [rbp-88h]
  _QWORD v35[2]; // [rsp+80h] [rbp-80h] BYREF
  __m128i *v36; // [rsp+90h] [rbp-70h] BYREF
  __int64 v37; // [rsp+98h] [rbp-68h]
  __m128i v38; // [rsp+A0h] [rbp-60h] BYREF
  _QWORD *v39; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v40; // [rsp+B8h] [rbp-48h]
  _QWORD v41[8]; // [rsp+C0h] [rbp-40h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F9F890) )
  {
    v5 = (_DWORD *)*a2;
    *a2 = 0;
    v6 = v5[2];
    if ( v6 == 10 )
    {
      if ( !byte_4FA5580 )
      {
LABEL_42:
        *a1 = 1;
        (*(void (__fastcall **)(_DWORD *))(*(_QWORD *)v5 + 8LL))(v5);
        return a1;
      }
    }
    else if ( (v6 & 0xFFFFFFFD) == 9 )
    {
      if ( byte_4FA54A0 )
        goto LABEL_42;
      v7 = **(_QWORD **)a3;
      if ( byte_4FA53C0 )
      {
        if ( *(_QWORD *)(v7 + 48) || (*(_BYTE *)(v7 + 32) & 0xF) == 1 )
          goto LABEL_42;
      }
LABEL_13:
      v11 = (char *)sub_1649960(v7);
      if ( v11 )
      {
        v39 = v41;
        sub_17E2210((__int64 *)&v39, v11, (__int64)&v11[v10]);
      }
      else
      {
        v40 = 0;
        v39 = v41;
        LOBYTE(v41[0]) = 0;
      }
      v33 = v35;
      sub_17E2210((__int64 *)&v33, " ", (__int64)"");
      (*(void (__fastcall **)(_QWORD **, _DWORD *))(*(_QWORD *)v5 + 24LL))(&v30, v5);
      v12 = 15;
      v13 = 15;
      if ( v30 != v32 )
        v13 = v32[0];
      v14 = v31 + v34;
      if ( v31 + v34 <= v13 )
        goto LABEL_21;
      if ( v33 != v35 )
        v12 = v35[0];
      if ( v14 <= v12 )
      {
        v15 = (__m128i *)sub_2241130(&v33, 0, 0, v30, v31);
        v36 = &v38;
        v16 = (__m128i *)v15->m128i_i64[0];
        v17 = v15 + 1;
        if ( (__m128i *)v15->m128i_i64[0] != &v15[1] )
          goto LABEL_22;
      }
      else
      {
LABEL_21:
        v15 = (__m128i *)sub_2241490(&v30, v33, v34, v14);
        v36 = &v38;
        v16 = (__m128i *)v15->m128i_i64[0];
        v17 = v15 + 1;
        if ( (__m128i *)v15->m128i_i64[0] != &v15[1] )
        {
LABEL_22:
          v36 = v16;
          v38.m128i_i64[0] = v15[1].m128i_i64[0];
LABEL_23:
          v37 = v15->m128i_i64[1];
          v15->m128i_i64[0] = (__int64)v17;
          v15->m128i_i64[1] = 0;
          v15[1].m128i_i8[0] = 0;
          v18 = 15;
          v19 = 15;
          if ( v36 != &v38 )
            v19 = v38.m128i_i64[0];
          v20 = v37 + v40;
          if ( v37 + v40 <= v19 )
            goto LABEL_29;
          if ( v39 != v41 )
            v18 = v41[0];
          if ( v20 <= v18 )
          {
            v25 = (__m128i *)sub_2241130(&v39, 0, 0, v36, v37);
            v27 = &v29;
            if ( (__m128i *)v25->m128i_i64[0] == &v25[1] )
            {
              v29 = _mm_loadu_si128(v25 + 1);
            }
            else
            {
              v27 = (__m128i *)v25->m128i_i64[0];
              v29.m128i_i64[0] = v25[1].m128i_i64[0];
            }
            v26 = v25->m128i_i64[1];
            v25[1].m128i_i8[0] = 0;
            v28 = v26;
            v25->m128i_i64[0] = (__int64)v25[1].m128i_i64;
            v25->m128i_i64[1] = 0;
          }
          else
          {
LABEL_29:
            v21 = (__m128i *)sub_2241490(&v36, v39, v40, v20);
            v27 = &v29;
            if ( (__m128i *)v21->m128i_i64[0] == &v21[1] )
            {
              v29 = _mm_loadu_si128(v21 + 1);
            }
            else
            {
              v27 = (__m128i *)v21->m128i_i64[0];
              v29.m128i_i64[0] = v21[1].m128i_i64[0];
            }
            v22 = v21->m128i_i64[1];
            v21[1].m128i_i8[0] = 0;
            v28 = v22;
            v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
            v21->m128i_i64[1] = 0;
          }
          if ( v36 != &v38 )
            j_j___libc_free_0(v36, v38.m128i_i64[0] + 1);
          if ( v30 != v32 )
            j_j___libc_free_0(v30, v32[0] + 1LL);
          if ( v33 != v35 )
            j_j___libc_free_0(v33, v35[0] + 1LL);
          if ( v39 != v41 )
            j_j___libc_free_0(v39, v41[0] + 1LL);
          v23 = *(_QWORD *)(a3 + 8);
          v38.m128i_i16[0] = 260;
          v36 = (__m128i *)&v27;
          v24 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a3 + 8LL) + 176LL);
          v40 = 0x100000012LL;
          v39 = &unk_49ECF40;
          v41[0] = v24;
          v41[1] = &v36;
          sub_16027F0(v23, (__int64)&v39);
          if ( v27 != &v29 )
            j_j___libc_free_0(v27, v29.m128i_i64[0] + 1);
          goto LABEL_42;
        }
      }
      v38 = _mm_loadu_si128(v15 + 1);
      goto LABEL_23;
    }
    v7 = **(_QWORD **)a3;
    goto LABEL_13;
  }
  v8 = *a2;
  *a2 = 0;
  *a1 = v8 | 1;
  return a1;
}
