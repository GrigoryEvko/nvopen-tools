// Function: sub_159E1A0
// Address: 0x159e1a0
//
__int64 __fastcall sub_159E1A0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8)
{
  __int64 v9; // r12
  __int64 *v10; // rbx
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // r15
  __m128i v15; // xmm0
  int v16; // r8d
  unsigned int v17; // r11d
  __int64 *v18; // rdi
  __int64 v19; // rax
  int v20; // r10d
  _QWORD *v21; // rdx
  __int64 v22; // r10
  _QWORD *v23; // rax
  int v25; // r11d
  _BYTE *v26; // r10
  unsigned int v27; // r11d
  _BYTE *v28; // rcx
  unsigned int v29; // r14d
  __int64 v30; // rdx
  __int64 *v31; // r13
  __int64 v32; // r9
  __int64 v33; // rbx
  __int64 v34; // r12
  __int64 *v35; // rdi
  _BYTE *v36; // rsi
  __int64 v37; // rax
  int v38; // eax
  int v39; // r8d
  unsigned int v40; // eax
  __int64 **v41; // rdx
  __int64 *v42; // rcx
  unsigned int v43; // eax
  __int64 v44; // r15
  __int64 v45; // r12
  __int64 v46; // r13
  __int64 *v47; // rdi
  unsigned int v48; // esi
  __int64 v49; // r8
  unsigned int v50; // edi
  __int64 **v51; // rdx
  __int64 *v52; // rax
  int v53; // edx
  int v54; // eax
  int v55; // eax
  int v56; // r11d
  _QWORD *v57; // rdx
  __int64 *v58; // rax
  __int64 v59; // r11
  int v60; // esi
  int v61; // [rsp+10h] [rbp-1E0h]
  int i; // [rsp+18h] [rbp-1D8h]
  _BYTE *v63; // [rsp+18h] [rbp-1D8h]
  __int64 v64; // [rsp+20h] [rbp-1D0h]
  __int64 v65; // [rsp+28h] [rbp-1C8h]
  _BYTE *v66; // [rsp+28h] [rbp-1C8h]
  int v67; // [rsp+28h] [rbp-1C8h]
  int v68; // [rsp+28h] [rbp-1C8h]
  int v69; // [rsp+30h] [rbp-1C0h]
  int v70; // [rsp+30h] [rbp-1C0h]
  __int64 v71; // [rsp+30h] [rbp-1C0h]
  __int64 **v72; // [rsp+30h] [rbp-1C0h]
  int v74; // [rsp+4Ch] [rbp-1A4h] BYREF
  __m128i v75; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v76; // [rsp+60h] [rbp-190h]
  __int64 v77[4]; // [rsp+70h] [rbp-180h] BYREF
  int v78; // [rsp+90h] [rbp-160h] BYREF
  __m128i v79; // [rsp+98h] [rbp-158h]
  __int64 v80; // [rsp+A8h] [rbp-148h]
  _BYTE *v81; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v82; // [rsp+B8h] [rbp-138h]
  _BYTE v83[304]; // [rsp+C0h] [rbp-130h] BYREF

  v9 = a1;
  v10 = a4;
  v11 = *a4;
  v75.m128i_i64[1] = (__int64)a2;
  v76 = a3;
  v75.m128i_i64[0] = v11;
  LODWORD(v81) = sub_1597240(a2, (__int64)&a2[a3]);
  v12 = sub_1597910(v75.m128i_i64, (int *)&v81);
  v13 = *(unsigned int *)(a1 + 24);
  v14 = *(_QWORD *)(a1 + 8);
  v15 = _mm_loadu_si128(&v75);
  v78 = v12;
  v80 = v76;
  v79 = v15;
  if ( !(_DWORD)v13 )
  {
LABEL_43:
    v41 = (__int64 **)(v14 + 8 * v13);
    goto LABEL_25;
  }
  v16 = v13 - 1;
  v17 = (v13 - 1) & v12;
  v18 = (__int64 *)(v14 + 8LL * v17);
  v19 = *v18;
  if ( *v18 == -8 )
    goto LABEL_15;
  for ( i = 1; ; ++i )
  {
    if ( v19 == -16 )
      goto LABEL_6;
    if ( v79.m128i_i64[0] != *(_QWORD *)v19 )
      goto LABEL_6;
    v20 = *(_DWORD *)(v19 + 20) & 0xFFFFFFF;
    if ( v76 != v20 )
      goto LABEL_6;
    if ( !v20 )
      break;
    v21 = (_QWORD *)v79.m128i_i64[1];
    v22 = v79.m128i_i64[1] + 8 + 8LL * (unsigned int)(v20 - 1);
    v23 = (_QWORD *)(-24 * v76 + v19);
    while ( *v21 == *v23 )
    {
      ++v21;
      v23 += 3;
      if ( (_QWORD *)v22 == v21 )
        goto LABEL_13;
    }
LABEL_6:
    v17 = v16 & (i + v17);
    v18 = (__int64 *)(v14 + 8LL * v17);
    v19 = *v18;
    if ( *v18 == -8 )
      goto LABEL_15;
  }
LABEL_13:
  if ( v18 != (__int64 *)(v14 + 8 * v13) )
    return *v18;
LABEL_15:
  v25 = *((_DWORD *)v10 + 5);
  v26 = v83;
  v81 = v83;
  v82 = 0x2000000000LL;
  v27 = v25 & 0xFFFFFFF;
  if ( v27 )
  {
    v28 = v83;
    v65 = v9;
    v29 = v27;
    v64 = a5;
    v30 = 0;
    v31 = v10;
    v32 = v10[-3 * v27];
    v33 = 1;
    v34 = v32;
    while ( 1 )
    {
      *(_QWORD *)&v28[8 * v30] = v34;
      v30 = (unsigned int)(v82 + 1);
      LODWORD(v82) = v82 + 1;
      if ( v29 == (_DWORD)v33 )
        break;
      v34 = v31[3 * (v33 - (*((_DWORD *)v31 + 5) & 0xFFFFFFF))];
      if ( HIDWORD(v82) <= (unsigned int)v30 )
      {
        v61 = v16;
        v63 = v26;
        sub_16CD150(&v81, v26, 0, 8);
        v30 = (unsigned int)v82;
        v16 = v61;
        v26 = v63;
      }
      v28 = v81;
      ++v33;
    }
    v35 = (__int64 *)v81;
    v10 = v31;
    v9 = v65;
    a5 = v64;
    v36 = &v81[8 * v30];
  }
  else
  {
    v36 = v83;
    v30 = 0;
    v35 = (__int64 *)v83;
  }
  v37 = *v10;
  v66 = v26;
  v69 = v16;
  v77[1] = (__int64)v35;
  v77[2] = v30;
  v77[0] = v37;
  v74 = sub_1597240(v35, (__int64)v36);
  v38 = sub_1597910(v77, &v74);
  v39 = v69;
  if ( v81 != v66 )
  {
    v67 = v69;
    v70 = v38;
    _libc_free((unsigned __int64)v81);
    v39 = v67;
    v38 = v70;
  }
  v40 = v39 & v38;
  v41 = (__int64 **)(v14 + 8LL * v40);
  v42 = *v41;
  if ( *v41 != v10 )
  {
    v53 = 1;
    while ( v42 != (__int64 *)-8LL )
    {
      v60 = v53 + 1;
      v40 = v39 & (v53 + v40);
      v41 = (__int64 **)(v14 + 8LL * v40);
      v42 = *v41;
      if ( *v41 == v10 )
        goto LABEL_25;
      v53 = v60;
    }
    v14 = *(_QWORD *)(v9 + 8);
    v13 = *(unsigned int *)(v9 + 24);
    goto LABEL_43;
  }
LABEL_25:
  *v41 = (__int64 *)-16LL;
  --*(_DWORD *)(v9 + 16);
  ++*(_DWORD *)(v9 + 20);
  if ( a7 == 1 )
  {
    sub_1593B40(&v10[3 * (a8 - (unsigned __int64)(*((_DWORD *)v10 + 5) & 0xFFFFFFF))], a6);
  }
  else
  {
    v43 = *((_DWORD *)v10 + 5) & 0xFFFFFFF;
    if ( v43 )
    {
      v71 = v9;
      v44 = 0;
      v45 = a5;
      v46 = v43 - 1;
      while ( 1 )
      {
        v47 = &v10[3 * (v44 - v43)];
        if ( v45 == *v47 )
          sub_1593B40(v47, a6);
        if ( v46 == v44 )
          break;
        ++v44;
        v43 = *((_DWORD *)v10 + 5) & 0xFFFFFFF;
      }
      v9 = v71;
    }
  }
  v48 = *(_DWORD *)(v9 + 24);
  if ( v48 )
  {
    v49 = *(_QWORD *)(v9 + 8);
    v50 = (v48 - 1) & v78;
    v51 = (__int64 **)(v49 + 8LL * v50);
    v52 = *v51;
    if ( *v51 != (__int64 *)-8LL )
    {
      v68 = 1;
      v72 = 0;
      while ( 1 )
      {
        if ( v52 == (__int64 *)-16LL )
        {
          if ( v72 )
            v51 = v72;
          v72 = v51;
        }
        else if ( v79.m128i_i64[0] == *v52 )
        {
          v56 = *((_DWORD *)v52 + 5) & 0xFFFFFFF;
          if ( v80 == v56 )
          {
            if ( !v56 )
              return 0;
            v57 = (_QWORD *)v79.m128i_i64[1];
            v58 = &v52[-3 * v80];
            v59 = v79.m128i_i64[1] + 8 + 8LL * (unsigned int)(v56 - 1);
            while ( *v57 == *v58 )
            {
              ++v57;
              v58 += 3;
              if ( v57 == (_QWORD *)v59 )
                return 0;
            }
          }
        }
        v50 = (v48 - 1) & (v68 + v50);
        v51 = (__int64 **)(v49 + 8LL * v50);
        v52 = *v51;
        if ( *v51 == (__int64 *)-8LL )
          break;
        ++v68;
      }
      if ( v72 )
        v51 = v72;
    }
    v54 = *(_DWORD *)(v9 + 16);
    ++*(_QWORD *)v9;
    v55 = v54 + 1;
    if ( 4 * v55 < 3 * v48 )
    {
      if ( v48 - *(_DWORD *)(v9 + 20) - v55 > v48 >> 3 )
        goto LABEL_48;
      goto LABEL_61;
    }
  }
  else
  {
    ++*(_QWORD *)v9;
  }
  v48 *= 2;
LABEL_61:
  sub_159DA60(v9, v48);
  sub_1598780(v9, (__int64)&v78, &v81);
  v51 = (__int64 **)v81;
  v55 = *(_DWORD *)(v9 + 16) + 1;
LABEL_48:
  *(_DWORD *)(v9 + 16) = v55;
  if ( *v51 != (__int64 *)-8LL )
    --*(_DWORD *)(v9 + 20);
  *v51 = v10;
  return 0;
}
