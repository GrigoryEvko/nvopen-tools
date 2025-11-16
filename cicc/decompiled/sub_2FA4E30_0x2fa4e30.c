// Function: sub_2FA4E30
// Address: 0x2fa4e30
//
_QWORD *__fastcall sub_2FA4E30(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  __int64 (*v9)(); // rax
  _QWORD *v10; // r15
  __int64 (*v11)(); // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r9
  int v17; // r8d
  unsigned int i; // eax
  __int64 v19; // rcx
  unsigned int v20; // eax
  __int64 (*v21)(); // rax
  __int64 (*v22)(); // rax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 *v28; // rax
  _QWORD *v29; // [rsp+8h] [rbp-238h]
  __int64 v30; // [rsp+20h] [rbp-220h] BYREF
  __int64 v31; // [rsp+A0h] [rbp-1A0h] BYREF
  __int64 v32; // [rsp+A8h] [rbp-198h]
  __int64 v33; // [rsp+B0h] [rbp-190h]
  __int64 v34; // [rsp+B8h] [rbp-188h]
  __int64 v35; // [rsp+C0h] [rbp-180h]
  __int64 *v36; // [rsp+C8h] [rbp-178h]
  __int64 v37; // [rsp+D0h] [rbp-170h]
  __int64 v38; // [rsp+D8h] [rbp-168h]
  __m128i v39; // [rsp+E0h] [rbp-160h] BYREF
  __m128i v40; // [rsp+F0h] [rbp-150h]
  __m128i v41; // [rsp+100h] [rbp-140h]
  __m128i v42; // [rsp+110h] [rbp-130h]
  __m128i v43; // [rsp+120h] [rbp-120h]
  __m128i v44; // [rsp+130h] [rbp-110h]
  __m128i v45; // [rsp+140h] [rbp-100h]
  __m128i v46; // [rsp+150h] [rbp-F0h]
  __m128i v47; // [rsp+160h] [rbp-E0h]
  __m128i v48; // [rsp+170h] [rbp-D0h]
  __int64 v49; // [rsp+180h] [rbp-C0h]
  __int64 v50; // [rsp+188h] [rbp-B8h]
  __int64 v51; // [rsp+190h] [rbp-B0h]
  __int64 v52; // [rsp+198h] [rbp-A8h]
  __int64 v53; // [rsp+1A0h] [rbp-A0h]
  __int64 v54; // [rsp+1A8h] [rbp-98h]
  _BYTE *v55; // [rsp+1B0h] [rbp-90h]
  __int64 v56; // [rsp+1B8h] [rbp-88h]
  _BYTE v57[64]; // [rsp+1C0h] [rbp-80h] BYREF
  __int64 v58; // [rsp+200h] [rbp-40h]

  v5 = *a2;
  v32 = 0;
  v33 = 0;
  v31 = v5;
  v34 = 0;
  v35 = 0;
  v37 = 0;
  v38 = 0;
  v49 = 0;
  v39 = _mm_loadu_si128(xmmword_3F8F0C0);
  v40 = _mm_loadu_si128(&xmmword_3F8F0C0[1]);
  v41 = _mm_loadu_si128(&xmmword_3F8F0C0[2]);
  v42 = _mm_loadu_si128(&xmmword_3F8F0C0[3]);
  v43 = _mm_loadu_si128(&xmmword_3F8F0C0[4]);
  v44 = v39;
  v45 = v40;
  v46 = v41;
  v47 = v42;
  v48 = v43;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = v57;
  v56 = 0x1000000000LL;
  v58 = 0;
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 16LL);
  if ( v6 == sub_23CE270 )
    BUG();
  v32 = ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, a3);
  v9 = *(__int64 (**)())(*(_QWORD *)v32 + 144LL);
  if ( v9 == sub_2C8F680 )
  {
    v33 = 0;
    BUG();
  }
  v10 = a1 + 4;
  v33 = ((__int64 (__fastcall *)(__int64))v9)(v32);
  v29 = a1 + 10;
  v11 = *(__int64 (**)())(*(_QWORD *)v33 + 104LL);
  if ( v11 != sub_2D56590 && !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v11)(v33, 0) )
  {
    v21 = *(__int64 (**)())(*(_QWORD *)v33 + 104LL);
    if ( v21 != sub_2D56590 && !((unsigned __int8 (__fastcall *)(__int64, __int64))v21)(v33, 1) )
    {
      v22 = *(__int64 (**)())(*(_QWORD *)v33 + 104LL);
      if ( v22 != sub_2D56590 && !((unsigned __int8 (__fastcall *)(__int64, __int64))v22)(v33, 2) )
        goto LABEL_15;
    }
  }
  v34 = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  if ( !(unsigned __int8)sub_DFAC40(v34) )
    goto LABEL_15;
  v12 = sub_BC1CD0(a4, &unk_4F82410, a3);
  v13 = *(_QWORD *)(a3 + 40);
  v14 = *(_QWORD *)(v12 + 8);
  v15 = *(unsigned int *)(v14 + 88);
  v16 = *(_QWORD *)(v14 + 72);
  if ( !(_DWORD)v15 )
    goto LABEL_20;
  v17 = 1;
  for ( i = (v15 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; i = (v15 - 1) & v20 )
  {
    v19 = v16 + 24LL * i;
    if ( *(_UNKNOWN **)v19 == &unk_4F87C68 && v13 == *(_QWORD *)(v19 + 8) )
      break;
    if ( *(_QWORD *)v19 == -4096 && *(_QWORD *)(v19 + 8) == -4096 )
      goto LABEL_20;
    v20 = v17 + i;
    ++v17;
  }
  if ( v19 == v16 + 24 * v15 )
  {
LABEL_20:
    v24 = 0;
  }
  else
  {
    v24 = *(_QWORD *)(*(_QWORD *)(v19 + 16) + 24LL);
    if ( v24 )
    {
      v24 += 8;
      v28 = &v30;
      do
      {
        *v28 = -4096;
        v28 += 2;
      }
      while ( v28 != &v31 );
    }
  }
  v37 = v24;
  v36 = (__int64 *)(sub_BC1CD0(a4, &unk_4F8D9A8, a3) + 8);
  if ( !(unsigned __int8)sub_11F2A60(a3, v37, v36)
    && (v35 = sub_BC1CD0(a4, &unk_4F875F0, a3) + 8,
        v38 = sub_BC1CD0(a4, &unk_4F8FAE8, a3) + 8,
        sub_2FF7BB0(&v39, v32),
        (unsigned __int8)sub_2FA48F0((__int64)&v31, a3, v25, v26, v27)) )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v10;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v29;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
LABEL_15:
    a1[1] = v10;
    a1[6] = 0;
    a1[7] = v29;
    a1[2] = 0x100000002LL;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  return a1;
}
