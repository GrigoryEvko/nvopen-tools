// Function: sub_3912580
// Address: 0x3912580
//
void __fastcall sub_3912580(__int64 a1, _QWORD *a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rbx
  __int64 v9; // r13
  signed __int64 v10; // rax
  char *v11; // rdx
  char *v12; // rax
  _BOOL8 v13; // rsi
  unsigned __int64 v14; // rbx
  unsigned int v15; // r10d
  char *v16; // rdx
  signed __int64 v17; // r12
  char *i; // rax
  char *v19; // r13
  __int64 v20; // rax
  void (*v21)(); // rcx
  __int64 v22; // rax
  unsigned int v23; // r10d
  void (*v24)(); // rcx
  __int64 v25; // rdx
  char v26; // al
  const char **v27; // rsi
  __int64 v28; // rsi
  unsigned __int64 v29; // r12
  __int64 v30; // rsi
  __int64 v31; // rsi
  signed __int64 v32; // rdx
  int v33; // eax
  unsigned __int64 v34; // rax
  __int64 v35; // [rsp+8h] [rbp-118h]
  void (*v36)(); // [rsp+20h] [rbp-100h]
  __int64 v37; // [rsp+28h] [rbp-F8h]
  unsigned int v38; // [rsp+30h] [rbp-F0h]
  char *v39; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v40; // [rsp+40h] [rbp-E0h]
  char *v42; // [rsp+48h] [rbp-D8h]
  char *v43; // [rsp+50h] [rbp-D0h] BYREF
  char *v44; // [rsp+58h] [rbp-C8h]
  __int64 v45; // [rsp+70h] [rbp-B0h]
  __m128i v46; // [rsp+90h] [rbp-90h] BYREF
  __int64 v47; // [rsp+A0h] [rbp-80h]
  const char *v48; // [rsp+B0h] [rbp-70h] BYREF
  char v49; // [rsp+C0h] [rbp-60h]
  char v50; // [rsp+C1h] [rbp-5Fh]
  __m128i v51; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v52; // [rsp+E0h] [rbp-40h]

  v8 = a2[1];
  v51.m128i_i64[0] = (__int64)"linetable_begin";
  LOWORD(v52) = 259;
  v9 = sub_38BF8E0(v8, (__int64)&v51, 0, 1);
  LOWORD(v52) = 259;
  v51.m128i_i64[0] = (__int64)"linetable_end";
  v35 = sub_38BF8E0(v8, (__int64)&v51, 0, 1);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 424LL))(a2, 242, 4);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a2 + 688LL))(a2, v35, v9, 4);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 176LL))(a2, v9, 0);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 328LL))(a2, a4, 0);
  (*(void (__fastcall **)(_QWORD *, __int64))(*a2 + 320LL))(a2, a4);
  sub_39122E0((unsigned __int64 *)&v43, a1, a3);
  v39 = v44;
  v40 = (unsigned __int64)v43;
  v10 = 0xAAAAAAAAAAAAAAABLL * ((v44 - v43) >> 3);
  if ( v10 >> 2 <= 0 )
  {
LABEL_58:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
        {
LABEL_61:
          v13 = 0;
          v40 = (unsigned __int64)v44;
          goto LABEL_9;
        }
LABEL_66:
        v34 = v40;
        if ( !*(_WORD *)(v40 + 12) )
          goto LABEL_61;
LABEL_67:
        v13 = v44 != (char *)v34;
        goto LABEL_9;
      }
      v34 = v40;
      if ( *(_WORD *)(v40 + 12) )
        goto LABEL_67;
      v40 += 24LL;
    }
    v34 = v40;
    if ( *(_WORD *)(v40 + 12) )
      goto LABEL_67;
    v40 += 24LL;
    goto LABEL_66;
  }
  v11 = v43;
  v12 = &v43[96 * (v10 >> 2)];
  while ( 1 )
  {
    if ( *((_WORD *)v11 + 6) )
      goto LABEL_8;
    if ( *((_WORD *)v11 + 18) )
    {
      v11 += 24;
LABEL_8:
      v40 = (unsigned __int64)v11;
      v13 = v44 != v11;
      goto LABEL_9;
    }
    if ( *((_WORD *)v11 + 30) )
    {
      v40 = (unsigned __int64)(v11 + 48);
      v13 = v44 != v11 + 48;
      goto LABEL_9;
    }
    if ( *((_WORD *)v11 + 42) )
      break;
    v11 += 96;
    if ( v12 == v11 )
    {
      v40 = (unsigned __int64)v11;
      v10 = 0xAAAAAAAAAAAAAAABLL * ((v44 - v11) >> 3);
      goto LABEL_58;
    }
  }
  v40 = (unsigned __int64)(v11 + 72);
  v13 = v44 != v11 + 72;
LABEL_9:
  (*(void (__fastcall **)(_QWORD *, _BOOL8, __int64))(*a2 + 424LL))(a2, v13, 2);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a2 + 688LL))(a2, a5, a4, 4);
  v14 = (unsigned __int64)v43;
  v42 = v44;
  if ( v43 != v44 )
  {
    while ( 1 )
    {
      v15 = *(_DWORD *)(v14 + 4);
      v16 = &v42[-v14];
      v17 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v42[-v14] >> 3);
      if ( v17 >> 2 > 0 )
        break;
      if ( v16 == (char *)48 )
      {
        v19 = (char *)v14;
        goto LABEL_51;
      }
      if ( v16 == (char *)72 )
      {
        v19 = (char *)v14;
        goto LABEL_56;
      }
LABEL_53:
      v19 = v42;
LABEL_19:
      v20 = *a2;
      v50 = 1;
      v49 = 3;
      v21 = *(void (**)())(v20 + 104);
      v38 = v15;
      v48 = "' begins";
      v36 = v21;
      v37 = *(unsigned int *)(*(_QWORD *)(a1 + 72) + 32LL * (v15 - 1));
      v22 = sub_390FEA0(a1);
      v23 = v38;
      v24 = v36;
      LOBYTE(v22) = *(_BYTE *)(*(_QWORD *)(v22 + 64) + v37);
      v25 = 2051;
      LOWORD(v47) = 2051;
      LOBYTE(v45) = v22;
      v46.m128i_i64[0] = (__int64)"Segment for file '";
      v46.m128i_i64[1] = v45;
      v26 = v49;
      if ( !v49 )
      {
        LOWORD(v52) = 256;
        if ( v36 == nullsub_580 )
          goto LABEL_25;
LABEL_38:
        ((void (__fastcall *)(_QWORD *, __m128i *, __int64))v36)(a2, &v51, 1);
        v23 = v38;
        goto LABEL_25;
      }
      if ( v49 == 1 )
      {
        v51 = _mm_loadu_si128(&v46);
        v52 = v47;
      }
      else
      {
        v27 = (const char **)v48;
        if ( v50 != 1 )
        {
          v27 = &v48;
          v26 = 2;
        }
        v25 = (__int64)&v46;
        v51.m128i_i64[1] = (__int64)v27;
        v51.m128i_i64[0] = (__int64)&v46;
        LOBYTE(v52) = 2;
        BYTE1(v52) = v26;
      }
      if ( v36 != nullsub_580 )
        goto LABEL_38;
LABEL_25:
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64, void (*)()))(*a2 + 672LL))(a2, v23, v25, v24);
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 424LL))(a2, (unsigned int)v17, 4);
      v28 = (unsigned int)(8 * v17 + 12);
      if ( (char *)v40 != v39 )
        v28 = (unsigned int)(v28 + 4 * v17);
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 424LL))(a2, v28, 4);
      if ( v19 != (char *)v14 )
      {
        v29 = v14;
        do
        {
          (*(void (__fastcall **)(_QWORD *, _QWORD, __int64, __int64))(*a2 + 688LL))(a2, *(_QWORD *)(v29 + 16), a4, 4);
          v30 = *(unsigned int *)(v29 + 8);
          if ( (*(_BYTE *)(v29 + 14) & 2) != 0 )
            v30 = *(_DWORD *)(v29 + 8) | 0x80000000;
          v29 += 24LL;
          (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 424LL))(a2, v30, 4);
        }
        while ( v19 != (char *)v29 );
        if ( (char *)v40 != v39 )
        {
          do
          {
            v31 = *(unsigned __int16 *)(v14 + 12);
            v14 += 24LL;
            (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 424LL))(a2, v31, 2);
            (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 424LL))(a2, 0, 2);
          }
          while ( v29 != v14 );
        }
      }
      if ( v19 == v42 )
        goto LABEL_46;
      v14 = (unsigned __int64)v19;
    }
    for ( i = (char *)v14; ; i += 96 )
    {
      if ( v15 != *((_DWORD *)i + 7) )
      {
        v19 = i + 24;
        goto LABEL_18;
      }
      if ( v15 != *((_DWORD *)i + 13) )
      {
        v19 = i + 48;
        goto LABEL_18;
      }
      if ( v15 != *((_DWORD *)i + 19) )
      {
        v19 = i + 72;
        goto LABEL_18;
      }
      v19 = i + 96;
      if ( (char *)(v14 + 96 * (v17 >> 2)) == i + 96 )
        break;
      if ( v15 != *((_DWORD *)i + 25) )
        goto LABEL_18;
    }
    v32 = v42 - v19;
    if ( v42 - v19 == 48 )
    {
      v33 = *((_DWORD *)i + 25);
    }
    else
    {
      if ( v32 != 72 )
      {
        if ( v32 != 24 )
        {
          v19 = v42;
          goto LABEL_19;
        }
        goto LABEL_52;
      }
      if ( v15 != *((_DWORD *)i + 25) )
      {
LABEL_18:
        v17 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v19[-v14] >> 3);
        goto LABEL_19;
      }
LABEL_56:
      v33 = *((_DWORD *)v19 + 7);
      v19 += 24;
    }
    if ( v15 != v33 )
      goto LABEL_18;
LABEL_51:
    v19 += 24;
LABEL_52:
    if ( v15 != *((_DWORD *)v19 + 1) )
      goto LABEL_18;
    goto LABEL_53;
  }
LABEL_46:
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 176LL))(a2, v35, 0);
  if ( v43 )
    j_j___libc_free_0((unsigned __int64)v43);
}
