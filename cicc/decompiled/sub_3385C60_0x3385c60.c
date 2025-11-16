// Function: sub_3385C60
// Address: 0x3385c60
//
char __fastcall sub_3385C60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rax
  char v6; // dl
  unsigned int v7; // r8d
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rax
  __int64 v12; // r14
  _DWORD *v13; // rax
  int v14; // eax
  unsigned __int8 **v15; // r12
  unsigned __int8 *v16; // rsi
  _DWORD *v17; // rax
  unsigned __int8 *v18; // rsi
  __int64 v19; // rbx
  _DWORD *v20; // r15
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned __int8 **v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // r10
  __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // eax
  int v30; // edx
  __int64 v31; // rsi
  int v32; // edi
  unsigned int v33; // esi
  __int64 v34; // r9
  unsigned int v35; // edx
  __int64 *v36; // rax
  __int64 v37; // r12
  int v38; // edi
  __int64 *v39; // rcx
  int v40; // edx
  int v41; // edx
  __int64 v42; // rbx
  char v44; // [rsp+3h] [rbp-13Dh]
  int v45; // [rsp+4h] [rbp-13Ch]
  __int64 v47; // [rsp+10h] [rbp-130h]
  char v48; // [rsp+10h] [rbp-130h]
  __int64 v49; // [rsp+10h] [rbp-130h]
  char v50; // [rsp+18h] [rbp-128h]
  unsigned __int64 v51; // [rsp+18h] [rbp-128h]
  __int64 v52; // [rsp+18h] [rbp-128h]
  _DWORD *v54; // [rsp+28h] [rbp-118h]
  unsigned __int64 v55; // [rsp+28h] [rbp-118h]
  _DWORD *v56; // [rsp+30h] [rbp-110h]
  __int64 v57; // [rsp+38h] [rbp-108h] BYREF
  __int64 *v58; // [rsp+48h] [rbp-F8h] BYREF
  __int64 *v59[2]; // [rsp+50h] [rbp-F0h] BYREF
  _DWORD *v60; // [rsp+60h] [rbp-E0h] BYREF
  __m128i v61; // [rsp+68h] [rbp-D8h] BYREF
  __int64 v62; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v63; // [rsp+88h] [rbp-B8h]
  __int64 v64; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v65; // [rsp+98h] [rbp-A8h]
  char v66; // [rsp+110h] [rbp-30h] BYREF

  v3 = &v64;
  v57 = a2;
  v62 = 0;
  v63 = 1;
  do
  {
    *v3 = -4096;
    v3 += 2;
  }
  while ( v3 != (__int64 *)&v66 );
  v4 = v57;
  v5 = *(_QWORD *)(*(_QWORD *)v57 + 104LL);
  v6 = v63 & 1;
  v45 = v5;
  v7 = 2 * v5;
  if ( 2 * (_DWORD)v5 )
  {
    ++v62;
    v8 = (8 * (int)v5 / 3u + 1) | ((unsigned __int64)(8 * (int)v5 / 3u + 1) >> 1);
    v9 = (((v8 >> 2) | v8) >> 4) | (v8 >> 2) | v8;
    v7 = ((((v9 >> 8) | v9) >> 16) | (v9 >> 8) | v9) + 1;
    v10 = 8;
    if ( v6 )
      goto LABEL_6;
  }
  else
  {
    ++v62;
    if ( v6 )
      goto LABEL_8;
  }
  v10 = v65;
LABEL_6:
  if ( v7 > v10 )
  {
    sub_3367360((__int64)&v62, v7);
    v4 = v57;
  }
LABEL_8:
  v59[1] = &v62;
  v59[0] = &v57;
  v11 = *(_QWORD *)(*(_QWORD *)v4 + 80LL);
  if ( !v11 )
    BUG();
  v12 = *(_QWORD *)(v11 + 32);
  v13 = (_DWORD *)(v11 + 24);
  v56 = v13;
  if ( (_DWORD *)v12 != v13 )
  {
    while ( 1 )
    {
      if ( !v12 )
        BUG();
      v14 = *(unsigned __int8 *)(v12 - 24);
      v15 = (unsigned __int8 **)(v12 - 24);
      if ( (_BYTE)v14 != 62 )
      {
        LODWORD(v13) = v14 - 67;
        if ( (unsigned int)v13 > 0xC )
        {
          LOBYTE(v13) = sub_B46AA0(v12 - 24);
          if ( !(_BYTE)v13 )
          {
            v13 = (_DWORD *)(32LL * (*(_DWORD *)(v12 - 20) & 0x7FFFFFF));
            if ( (*(_BYTE *)(v12 - 17) & 0x40) != 0 )
            {
              v24 = *(unsigned __int8 ***)(v12 - 32);
              v15 = (unsigned __int8 **)((char *)v13 + (_QWORD)v24);
            }
            else
            {
              v24 = (unsigned __int8 **)((char *)v15 - (char *)v13);
            }
            for ( ; v15 != v24; v24 += 4 )
            {
              v13 = (_DWORD *)sub_3368280(v59, *v24);
              if ( v13 )
                *v13 = 1;
            }
          }
        }
        goto LABEL_21;
      }
      v16 = *(unsigned __int8 **)(v12 - 88);
      v17 = (_DWORD *)sub_3368280(v59, v16);
      if ( v17 )
        *v17 = 1;
      v18 = sub_BD3990(*(unsigned __int8 **)(v12 - 56), (__int64)v16);
      v19 = (__int64)v18;
      v13 = (_DWORD *)sub_3368280(v59, v18);
      v20 = v13;
      if ( v13 )
      {
        LODWORD(v13) = *v13;
        if ( !(_DWORD)v13 )
          break;
      }
LABEL_21:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v56 == (_DWORD *)v12 )
        goto LABEL_22;
    }
    v13 = sub_BD3990(*(unsigned __int8 **)(v12 - 88), (__int64)v18);
    if ( *(_BYTE *)v13 != 22 )
      goto LABEL_20;
    v54 = v13;
    LOBYTE(v13) = sub_B2BAE0((__int64)v13);
    if ( (_BYTE)v13 )
      goto LABEL_20;
    LOBYTE(v13) = sub_BCADB0(*((_QWORD *)v54 + 1));
    if ( (_BYTE)v13 )
      goto LABEL_20;
    v47 = *((_QWORD *)v18 + 9);
    v50 = sub_AE5020(a1, v47);
    v21 = sub_9208B0(a1, v47);
    v61.m128i_i64[0] = v22;
    v60 = (_DWORD *)v21;
    v48 = v22;
    v51 = ((1LL << v50) + ((unsigned __int64)(v21 + 7) >> 3) - 1) >> v50 << v50;
    v60 = (_DWORD *)sub_9208B0(a1, *((_QWORD *)v54 + 1));
    v13 = (_DWORD *)(((unsigned __int64)v60 + 7) >> 3);
    v61.m128i_i64[0] = v23;
    if ( v13 != (_DWORD *)v51 )
      goto LABEL_20;
    if ( v48 != (_BYTE)v23 )
      goto LABEL_20;
    v49 = (__int64)v54;
    v52 = *((_QWORD *)v54 + 1);
    v60 = (_DWORD *)sub_9208B0(a1, v52);
    v61.m128i_i64[0] = v25;
    v55 = ((unsigned __int64)v60 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v44 = v25;
    v13 = (_DWORD *)sub_9208B0(a1, v52);
    v26 = v49;
    v60 = v13;
    v61.m128i_i64[0] = v27;
    if ( v13 != (_DWORD *)v55 || (LOBYTE(v13) = v44, v61.m128i_i8[0] != v44) )
    {
LABEL_20:
      *v20 = 1;
      goto LABEL_21;
    }
    v28 = *(_QWORD *)(a3 + 8);
    v29 = *(_DWORD *)(a3 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      LODWORD(v13) = (v29 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
      v31 = *(_QWORD *)(v28 + 24LL * (unsigned int)v13);
      if ( v49 == v31 )
        goto LABEL_20;
      v32 = 1;
      while ( v31 != -4096 )
      {
        LODWORD(v13) = v30 & (v32 + (_DWORD)v13);
        v31 = *(_QWORD *)(v28 + 24LL * (unsigned int)v13);
        if ( v49 == v31 )
          goto LABEL_20;
        ++v32;
      }
    }
    *v20 = 2;
    v60 = (_DWORD *)v49;
    v33 = *(_DWORD *)(a3 + 24);
    v61.m128i_i64[0] = v19;
    v61.m128i_i64[1] = v12 - 24;
    if ( v33 )
    {
      v34 = *(_QWORD *)(a3 + 8);
      v35 = (v33 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
      v36 = (__int64 *)(v34 + 24LL * v35);
      v37 = *v36;
      if ( v49 == *v36 )
        goto LABEL_44;
      v38 = 1;
      v39 = 0;
      while ( v37 != -4096 )
      {
        if ( v37 == -8192 && !v39 )
          v39 = v36;
        v35 = (v33 - 1) & (v38 + v35);
        v36 = (__int64 *)(v34 + 24LL * v35);
        v37 = *v36;
        if ( v49 == *v36 )
          goto LABEL_44;
        ++v38;
      }
      if ( v39 )
        v36 = v39;
      ++*(_QWORD *)a3;
      v40 = *(_DWORD *)(a3 + 16);
      v58 = v36;
      v41 = v40 + 1;
      if ( 4 * v41 < 3 * v33 )
      {
        v42 = a3;
        if ( v33 - *(_DWORD *)(a3 + 20) - v41 > v33 >> 3 )
        {
LABEL_54:
          *(_DWORD *)(a3 + 16) = v41;
          if ( *v36 != -4096 )
            --*(_DWORD *)(a3 + 20);
          *v36 = v26;
          *(__m128i *)(v36 + 1) = _mm_loadu_si128(&v61);
LABEL_44:
          LOBYTE(v13) = a3;
          if ( v45 == *(_DWORD *)(a3 + 16) )
            goto LABEL_22;
          goto LABEL_21;
        }
LABEL_59:
        sub_3385A60(v42, v33);
        sub_337D1B0(v42, (__int64 *)&v60, &v58);
        v26 = (__int64)v60;
        v41 = *(_DWORD *)(v42 + 16) + 1;
        v36 = v58;
        goto LABEL_54;
      }
    }
    else
    {
      v58 = 0;
      ++*(_QWORD *)a3;
    }
    v42 = a3;
    v33 *= 2;
    goto LABEL_59;
  }
LABEL_22:
  if ( (v63 & 1) == 0 )
    LOBYTE(v13) = sub_C7D6A0(v64, 16LL * v65, 8);
  return (char)v13;
}
