// Function: sub_AD2630
// Address: 0xad2630
//
__int64 __fastcall sub_AD2630(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8)
{
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // r15
  __m128i v15; // xmm0
  int v16; // r8d
  unsigned int v17; // edx
  __int64 *v18; // r9
  __int64 v19; // rax
  int v20; // r11d
  _QWORD *v21; // rcx
  __int64 v22; // r11
  _QWORD *v23; // rax
  int v25; // ecx
  __int64 *v26; // r10
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r14
  __int64 v31; // r13
  __int64 v32; // rbx
  __int64 v33; // r12
  int v34; // eax
  int v35; // r8d
  unsigned int v36; // eax
  __int64 *v37; // rdx
  __int64 v38; // rcx
  unsigned int v39; // eax
  __int64 v40; // r15
  __int64 v41; // r12
  __int64 v42; // r13
  _QWORD *v43; // rdi
  unsigned int v44; // esi
  __int64 v45; // rdi
  unsigned int v46; // ecx
  __int64 *v47; // rdx
  __int64 v48; // rax
  int v49; // eax
  int v50; // eax
  int v51; // edx
  int v52; // r13d
  _QWORD *v53; // rdx
  _QWORD *v54; // rax
  __int64 v55; // r13
  int v56; // esi
  __int64 *v57; // [rsp+0h] [rbp-1E0h]
  int i; // [rsp+Ch] [rbp-1D4h]
  int v59; // [rsp+Ch] [rbp-1D4h]
  __int64 v60; // [rsp+10h] [rbp-1D0h]
  __int64 *v61; // [rsp+18h] [rbp-1C8h]
  int v62; // [rsp+18h] [rbp-1C8h]
  int v63; // [rsp+18h] [rbp-1C8h]
  __int64 *v65; // [rsp+20h] [rbp-1C0h]
  int v66; // [rsp+28h] [rbp-1B8h]
  int v67; // [rsp+28h] [rbp-1B8h]
  __int64 v68; // [rsp+28h] [rbp-1B8h]
  int v69; // [rsp+3Ch] [rbp-1A4h] BYREF
  __m128i v70; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 v71; // [rsp+50h] [rbp-190h]
  __int64 v72[4]; // [rsp+60h] [rbp-180h] BYREF
  int v73; // [rsp+80h] [rbp-160h] BYREF
  __m128i v74; // [rsp+88h] [rbp-158h]
  __int64 v75; // [rsp+98h] [rbp-148h]
  __int64 *v76; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v77; // [rsp+A8h] [rbp-138h]
  _BYTE v78[304]; // [rsp+B0h] [rbp-130h] BYREF

  v9 = a1;
  v10 = a4;
  v11 = *(_QWORD *)(a4 + 8);
  v70.m128i_i64[1] = (__int64)a2;
  v71 = a3;
  v70.m128i_i64[0] = v11;
  LODWORD(v76) = sub_AC5F60(a2, (__int64)&a2[a3]);
  v12 = sub_AC7520(v70.m128i_i64, &v76);
  v13 = *(unsigned int *)(a1 + 24);
  v14 = *(_QWORD *)(a1 + 8);
  v15 = _mm_loadu_si128(&v70);
  v73 = v12;
  v75 = v71;
  v74 = v15;
  if ( !(_DWORD)v13 )
  {
LABEL_54:
    v37 = (__int64 *)(v14 + 8 * v13);
    goto LABEL_28;
  }
  v16 = v13 - 1;
  v17 = (v13 - 1) & v12;
  v18 = (__int64 *)(v14 + 8LL * v17);
  v19 = *v18;
  if ( *v18 == -4096 )
    goto LABEL_16;
  for ( i = 1; ; ++i )
  {
    if ( v19 == -8192 )
      goto LABEL_6;
    if ( v74.m128i_i64[0] != *(_QWORD *)(v19 + 8) )
      goto LABEL_6;
    v20 = *(_DWORD *)(v19 + 4) & 0x7FFFFFF;
    if ( v71 != v20 )
      goto LABEL_6;
    if ( !v20 )
      break;
    v21 = (_QWORD *)v74.m128i_i64[1];
    v22 = v74.m128i_i64[1] + 8 + 8LL * (unsigned int)(v20 - 1);
    v23 = (_QWORD *)(-32 * v71 + v19);
    while ( *v21 == *v23 )
    {
      ++v21;
      v23 += 4;
      if ( (_QWORD *)v22 == v21 )
        goto LABEL_14;
    }
LABEL_6:
    v17 = v16 & (i + v17);
    v18 = (__int64 *)(v14 + 8LL * v17);
    v19 = *v18;
    if ( *v18 == -4096 )
      goto LABEL_16;
  }
LABEL_14:
  if ( v18 != (__int64 *)(v14 + 8 * v13) )
    return *v18;
LABEL_16:
  v25 = *(_DWORD *)(v10 + 4);
  v26 = (__int64 *)v78;
  v27 = 0;
  v77 = 0x2000000000LL;
  v28 = 0;
  v29 = v25 & 0x7FFFFFF;
  v76 = (__int64 *)v78;
  if ( (unsigned int)v29 > 0x20uLL )
  {
    sub_C8D5F0(&v76, v78, (unsigned int)v29, 8);
    v28 = (unsigned int)v77;
    v16 = v13 - 1;
    v26 = (__int64 *)v78;
    v29 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    v27 = (unsigned int)v77;
  }
  if ( (_DWORD)v29 )
  {
    v60 = a5;
    v30 = (unsigned int)(v29 - 1);
    v31 = v10;
    v32 = 0;
    v59 = v16;
    while ( 1 )
    {
      v33 = *(_QWORD *)(v31 + 32 * (v32 - v29));
      if ( v27 + 1 > (unsigned __int64)HIDWORD(v77) )
      {
        v57 = v26;
        sub_C8D5F0(&v76, v26, v27 + 1, 8);
        v27 = (unsigned int)v77;
        v26 = v57;
      }
      v76[v27] = v33;
      v27 = (unsigned int)(v77 + 1);
      LODWORD(v77) = v77 + 1;
      if ( v30 == v32 )
        break;
      ++v32;
      v29 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
    }
    v10 = v31;
    v9 = a1;
    v28 = (unsigned int)v27;
    a5 = v60;
    v16 = v59;
  }
  v61 = v26;
  v66 = v16;
  v72[0] = *(_QWORD *)(v10 + 8);
  v72[1] = (__int64)v76;
  v72[2] = v28;
  v69 = sub_AC5F60(v76, (__int64)&v76[v28]);
  v34 = sub_AC7520(v72, &v69);
  v35 = v66;
  if ( v76 != v61 )
  {
    v62 = v66;
    v67 = v34;
    _libc_free(v76, &v69);
    v35 = v62;
    v34 = v67;
  }
  v36 = v35 & v34;
  v37 = (__int64 *)(v14 + 8LL * v36);
  v38 = *v37;
  if ( *v37 != v10 )
  {
    v51 = 1;
    while ( v38 != -4096 )
    {
      v56 = v51 + 1;
      v36 = v35 & (v51 + v36);
      v37 = (__int64 *)(v14 + 8LL * v36);
      v38 = *v37;
      if ( *v37 == v10 )
        goto LABEL_28;
      v51 = v56;
    }
    v14 = *(_QWORD *)(v9 + 8);
    v13 = *(unsigned int *)(v9 + 24);
    goto LABEL_54;
  }
LABEL_28:
  *v37 = -8192;
  --*(_DWORD *)(v9 + 16);
  ++*(_DWORD *)(v9 + 20);
  if ( a7 == 1 )
  {
    sub_AC2B30(v10 + 32 * (a8 - (unsigned __int64)(*(_DWORD *)(v10 + 4) & 0x7FFFFFF)), a6);
  }
  else
  {
    v39 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    if ( v39 )
    {
      v68 = v9;
      v40 = 0;
      v41 = a5;
      v42 = v39 - 1;
      while ( 1 )
      {
        v43 = (_QWORD *)(v10 + 32 * (v40 - v39));
        if ( v41 == *v43 )
          sub_AC2B30((__int64)v43, a6);
        if ( v42 == v40 )
          break;
        ++v40;
        v39 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
      }
      v9 = v68;
    }
  }
  v44 = *(_DWORD *)(v9 + 24);
  if ( v44 )
  {
    v45 = *(_QWORD *)(v9 + 8);
    v46 = (v44 - 1) & v73;
    v47 = (__int64 *)(v45 + 8LL * v46);
    v48 = *v47;
    if ( *v47 != -4096 )
    {
      v63 = 1;
      v65 = 0;
      while ( 1 )
      {
        if ( v48 == -8192 )
        {
          if ( v65 )
            v47 = v65;
          v65 = v47;
        }
        else if ( v74.m128i_i64[0] == *(_QWORD *)(v48 + 8) )
        {
          v52 = *(_DWORD *)(v48 + 4) & 0x7FFFFFF;
          if ( v75 == v52 )
          {
            if ( !v52 )
              return 0;
            v53 = (_QWORD *)v74.m128i_i64[1];
            v54 = (_QWORD *)(-32 * v75 + v48);
            v55 = v74.m128i_i64[1] + 8 + 8LL * (unsigned int)(v52 - 1);
            while ( *v53 == *v54 )
            {
              ++v53;
              v54 += 4;
              if ( (_QWORD *)v55 == v53 )
                return 0;
            }
          }
        }
        v46 = (v44 - 1) & (v63 + v46);
        v47 = (__int64 *)(v45 + 8LL * v46);
        v48 = *v47;
        if ( *v47 == -4096 )
          break;
        ++v63;
      }
      if ( v65 )
        v47 = v65;
    }
    v49 = *(_DWORD *)(v9 + 16);
    ++*(_QWORD *)v9;
    v76 = v47;
    v50 = v49 + 1;
    if ( 4 * v50 < 3 * v44 )
    {
      if ( v44 - *(_DWORD *)(v9 + 20) - v50 > v44 >> 3 )
        goto LABEL_47;
      goto LABEL_64;
    }
  }
  else
  {
    ++*(_QWORD *)v9;
    v76 = 0;
  }
  v44 *= 2;
LABEL_64:
  sub_AD1D70(v9, v44);
  sub_AC8130(v9, (__int64)&v73, &v76);
  v47 = v76;
  v50 = *(_DWORD *)(v9 + 16) + 1;
LABEL_47:
  *(_DWORD *)(v9 + 16) = v50;
  if ( *v47 != -4096 )
    --*(_DWORD *)(v9 + 20);
  *v47 = v10;
  return 0;
}
