// Function: sub_373CB80
// Address: 0x373cb80
//
void __fastcall sub_373CB80(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int *a5, __int64 a6)
{
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // r10
  __int64 v10; // r9
  __int64 v11; // rbx
  __int64 v12; // r11
  unsigned int v13; // r12d
  unsigned int v14; // r13d
  bool v15; // al
  __int64 v16; // r14
  unsigned int v17; // esi
  int v18; // r15d
  unsigned int i; // ecx
  unsigned int v20; // ecx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // r15
  __int64 v24; // r14
  bool v25; // zf
  __int64 v26; // rax
  char *v27; // rax
  int v28; // eax
  __int64 v29; // rax
  int v30; // edi
  int v31; // edi
  unsigned int *v32; // rcx
  int v33; // edx
  unsigned int v34; // eax
  unsigned int v35; // eax
  unsigned int v36; // edx
  int v37; // eax
  int v38; // edx
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 v41; // rcx
  const __m128i *v42; // rdx
  __m128i *v43; // rax
  unsigned __int64 v44; // r12
  __int64 v45; // rdi
  const void *v46; // rsi
  int v47; // esi
  int v48; // esi
  __int64 v49; // rdi
  unsigned int *v50; // rax
  unsigned int v51; // r15d
  int v52; // edx
  unsigned int v53; // ecx
  unsigned int v54; // r15d
  unsigned int v55; // ecx
  unsigned int v56; // esi
  __int64 *v58; // [rsp+18h] [rbp-D8h]
  int v59; // [rsp+28h] [rbp-C8h]
  __int64 v60; // [rsp+28h] [rbp-C8h]
  __int64 v61; // [rsp+28h] [rbp-C8h]
  __int64 v62; // [rsp+28h] [rbp-C8h]
  unsigned int *v63; // [rsp+30h] [rbp-C0h]
  __int64 v64; // [rsp+30h] [rbp-C0h]
  __int64 v65; // [rsp+30h] [rbp-C0h]
  __int64 v66; // [rsp+30h] [rbp-C0h]
  __int64 v67; // [rsp+30h] [rbp-C0h]
  __int64 *v68; // [rsp+38h] [rbp-B8h]
  __int64 v69; // [rsp+40h] [rbp-B0h]
  __int64 v70; // [rsp+40h] [rbp-B0h]
  unsigned int *v71; // [rsp+40h] [rbp-B0h]
  __int64 v72; // [rsp+48h] [rbp-A8h]
  __int64 v73; // [rsp+50h] [rbp-A0h]
  char *v75; // [rsp+60h] [rbp-90h] BYREF
  __int64 v76; // [rsp+68h] [rbp-88h]
  _BYTE v77[32]; // [rsp+70h] [rbp-80h] BYREF
  _QWORD *v78; // [rsp+90h] [rbp-60h] BYREF
  __int64 v79; // [rsp+98h] [rbp-58h]
  _QWORD v80[10]; // [rsp+A0h] [rbp-50h] BYREF

  v7 = *(unsigned int *)(a3 + 8);
  v75 = v77;
  v76 = 0x200000000LL;
  if ( v7 > 2 )
  {
    sub_C8D5F0((__int64)&v75, v77, v7, 0x10u, (__int64)a5, a6);
    v7 = *(unsigned int *)(a3 + 8);
  }
  v8 = 16 * v7;
  v58 = (__int64 *)(*(_QWORD *)a3 + v8);
  if ( *(__int64 **)a3 == v58 )
  {
    v28 = v76;
    goto LABEL_27;
  }
  v68 = *(__int64 **)a3;
  do
  {
    v72 = sub_3211F40(a1[26], *v68);
    v73 = sub_3211FB0(a1[26], v68[1]);
    v9 = *(_QWORD *)(*v68 + 24);
    v10 = *(_QWORD *)(v68[1] + 24);
    v11 = v9;
    LODWORD(a4) = *(_DWORD *)(v10 + 252);
    LODWORD(v8) = *(_DWORD *)(v10 + 256);
    v12 = v10;
    while ( 2 )
    {
      v13 = *(_DWORD *)(v11 + 252);
      v14 = *(_DWORD *)(v11 + 256);
      v15 = v13 == (_DWORD)a4 && v14 == (_DWORD)v8;
      if ( !v15 && !*(_BYTE *)(v11 + 261) )
        goto LABEL_23;
      v16 = a1[23];
      v17 = *(_DWORD *)(v16 + 328);
      v69 = v16 + 304;
      if ( !v17 )
      {
        ++*(_QWORD *)(v16 + 304);
        goto LABEL_35;
      }
      v59 = 1;
      a6 = *(_QWORD *)(v16 + 312);
      v63 = 0;
      v18 = ((0xBF58476D1CE4E5B9LL * ((37 * v14) | ((unsigned __int64)(37 * v13) << 32))) >> 31) ^ (756364221 * v14);
      for ( i = v18 & (v17 - 1); ; i = (v17 - 1) & v20 )
      {
        a5 = (unsigned int *)(a6 + 12LL * i);
        if ( v13 == *a5 && v14 == a5[1] )
        {
          v21 = a5[2];
          goto LABEL_15;
        }
        if ( !*a5 )
          break;
LABEL_12:
        v20 = v59 + i;
        ++v59;
      }
      v36 = a5[1];
      if ( v36 != -1 )
      {
        if ( v36 == -2 )
        {
          if ( v63 )
            a5 = v63;
          v63 = a5;
        }
        goto LABEL_12;
      }
      if ( v63 )
        a5 = v63;
      v37 = *(_DWORD *)(v16 + 320);
      ++*(_QWORD *)(v16 + 304);
      v38 = v37 + 1;
      if ( 4 * (v37 + 1) < 3 * v17 )
      {
        if ( v17 - *(_DWORD *)(v16 + 324) - v38 > v17 >> 3 )
          goto LABEL_53;
        v62 = v9;
        v67 = v12;
        sub_31E5EB0(v69, v17);
        v47 = *(_DWORD *)(v16 + 328);
        if ( !v47 )
        {
LABEL_86:
          ++*(_DWORD *)(v16 + 320);
          BUG();
        }
        v48 = v47 - 1;
        v50 = 0;
        v12 = v67;
        v9 = v62;
        v51 = v48 & v18;
        v52 = 1;
        while ( 2 )
        {
          v49 = *(_QWORD *)(v16 + 312);
          a5 = (unsigned int *)(v49 + 12LL * v51);
          v53 = *a5;
          if ( v13 == *a5 )
          {
            if ( v14 == a5[1] )
              goto LABEL_46;
            if ( v53 )
              goto LABEL_65;
          }
          else if ( v53 )
          {
LABEL_65:
            v54 = v52 + v51;
            ++v52;
            v51 = v48 & v54;
            continue;
          }
          break;
        }
        v55 = a5[1];
        if ( v55 == -1 )
        {
          if ( v50 )
            a5 = v50;
          v38 = *(_DWORD *)(v16 + 320) + 1;
          goto LABEL_53;
        }
        if ( !v50 && v55 == -2 )
          v50 = (unsigned int *)(v49 + 12LL * v51);
        goto LABEL_65;
      }
LABEL_35:
      v60 = v9;
      v65 = v12;
      sub_31E5EB0(v69, 2 * v17);
      v30 = *(_DWORD *)(v16 + 328);
      if ( !v30 )
        goto LABEL_86;
      v31 = v30 - 1;
      v32 = 0;
      a6 = *(_QWORD *)(v16 + 312);
      v12 = v65;
      v9 = v60;
      v33 = 1;
      v34 = v31
          & (((0xBF58476D1CE4E5B9LL * ((37 * v14) | ((unsigned __int64)(37 * v13) << 32))) >> 31)
           ^ (756364221 * v14));
      while ( 2 )
      {
        a5 = (unsigned int *)(a6 + 12LL * v34);
        if ( v13 == *a5 && v14 == a5[1] )
        {
LABEL_46:
          ++*(_DWORD *)(v16 + 320);
          if ( *a5 )
            goto LABEL_54;
          goto LABEL_47;
        }
        if ( *a5 )
        {
LABEL_39:
          v35 = v33 + v34;
          ++v33;
          v34 = v31 & v35;
          continue;
        }
        break;
      }
      v56 = a5[1];
      if ( v56 != -1 )
      {
        if ( v56 == -2 && !v32 )
          v32 = (unsigned int *)(a6 + 12LL * v34);
        goto LABEL_39;
      }
      if ( v32 )
        a5 = v32;
      v38 = *(_DWORD *)(v16 + 320) + 1;
LABEL_53:
      *(_DWORD *)(v16 + 320) = v38;
      if ( *a5 )
        goto LABEL_54;
LABEL_47:
      if ( a5[1] == -1 )
        goto LABEL_55;
LABEL_54:
      --*(_DWORD *)(v16 + 324);
LABEL_55:
      *a5 = v13;
      a5[1] = v14;
      a5[2] = 0;
      v78 = (_QWORD *)__PAIR64__(v14, v13);
      v79 = 0;
      v80[0] = 0;
      v39 = *(unsigned int *)(v16 + 344);
      v40 = v39 + 1;
      if ( v39 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 348) )
      {
        v44 = *(_QWORD *)(v16 + 336);
        v61 = v9;
        v45 = v16 + 336;
        v66 = v12;
        v46 = (const void *)(v16 + 352);
        v71 = a5;
        if ( v44 > (unsigned __int64)&v78 || (unsigned __int64)&v78 >= v44 + 24 * v39 )
        {
          sub_C8D5F0(v45, v46, v40, 0x18u, (__int64)a5, a6);
          v41 = *(_QWORD *)(v16 + 336);
          v39 = *(unsigned int *)(v16 + 344);
          v42 = (const __m128i *)&v78;
          v9 = v61;
          v12 = v66;
          a5 = v71;
        }
        else
        {
          sub_C8D5F0(v45, v46, v40, 0x18u, (__int64)a5, a6);
          v41 = *(_QWORD *)(v16 + 336);
          v39 = *(unsigned int *)(v16 + 344);
          a5 = v71;
          v12 = v66;
          v9 = v61;
          v42 = (const __m128i *)((char *)&v78 + v41 - v44);
        }
      }
      else
      {
        v41 = *(_QWORD *)(v16 + 336);
        v42 = (const __m128i *)&v78;
      }
      v43 = (__m128i *)(v41 + 24 * v39);
      *v43 = _mm_loadu_si128(v42);
      v43[1].m128i_i64[0] = v42[1].m128i_i64[0];
      v21 = *(unsigned int *)(v16 + 344);
      *(_DWORD *)(v16 + 344) = v21 + 1;
      a5[2] = v21;
      v13 = *(_DWORD *)(v11 + 252);
      v14 = *(_DWORD *)(v11 + 256);
      v15 = *(_DWORD *)(v12 + 252) == v13 && *(_DWORD *)(v12 + 256) == v14;
LABEL_15:
      v22 = *(_QWORD *)(v16 + 336) + 24 * v21;
      v23 = *(_QWORD *)(v22 + 8);
      v24 = *(_QWORD *)(v22 + 16);
      if ( *(_DWORD *)(v9 + 256) == v14 && *(_DWORD *)(v9 + 252) == v13 )
        v23 = v72;
      v25 = !v15;
      v26 = (unsigned int)v76;
      if ( !v25 )
        v24 = v73;
      if ( (unsigned __int64)(unsigned int)v76 + 1 > HIDWORD(v76) )
      {
        v64 = v9;
        v70 = v12;
        sub_C8D5F0((__int64)&v75, v77, (unsigned int)v76 + 1LL, 0x10u, (__int64)a5, a6);
        v26 = (unsigned int)v76;
        v9 = v64;
        v12 = v70;
      }
      v27 = &v75[16 * v26];
      *(_QWORD *)v27 = v23;
      *((_QWORD *)v27 + 1) = v24;
      a4 = *(unsigned int *)(v12 + 252);
      v8 = *(unsigned int *)(v12 + 256);
      v28 = v76 + 1;
      LODWORD(v76) = v76 + 1;
      if ( *(_QWORD *)(v11 + 252) != __PAIR64__(v8, a4) )
      {
LABEL_23:
        v29 = *(_QWORD *)(v11 + 32);
        v11 = *(_QWORD *)(v11 + 8);
        if ( v29 + 320 == v11 )
          v11 = 0;
        continue;
      }
      break;
    }
    v68 += 2;
  }
  while ( v58 != v68 );
LABEL_27:
  v78 = v80;
  v79 = 0x200000000LL;
  if ( v28 )
    sub_37352C0((__int64)&v78, &v75, v8, a4, (__int64)a5, a6);
  sub_3739060(a1, a2, (char *)&v78, a4, (__int64)a5, a6);
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
  if ( v75 != v77 )
    _libc_free((unsigned __int64)v75);
}
