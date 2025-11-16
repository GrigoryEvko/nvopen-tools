// Function: sub_7987E0
// Address: 0x7987e0
//
__int64 __fastcall sub_7987E0(__int64 a1, __int64 a2, __int64 i, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  unsigned __int64 v8; // rbx
  __int64 v9; // rsi
  unsigned int v10; // eax
  int v11; // edi
  int v12; // eax
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // r15
  _QWORD *v15; // rax
  unsigned int v16; // eax
  unsigned __int64 j; // rbx
  __int64 v18; // rax
  char k; // dl
  unsigned int v20; // edi
  int v21; // esi
  __int64 v22; // rcx
  __int64 *v23; // r12
  unsigned int v24; // edx
  _DWORD *n; // rax
  __int64 v26; // rcx
  int v27; // edi
  __int64 v28; // rsi
  unsigned int v29; // eax
  __int64 v30; // rdx
  unsigned __int64 v32; // r12
  char v33; // al
  int v34; // r9d
  unsigned int v35; // edx
  unsigned int v36; // eax
  unsigned int v37; // edx
  __int64 v38; // r13
  char *v39; // rdi
  __int64 v40; // rax
  _QWORD *v41; // r13
  unsigned int v42; // edx
  __int64 v43; // rsi
  unsigned int v44; // ecx
  __m128i *v45; // rax
  __m128i v46; // xmm0
  __m128i *v47; // rax
  int v48; // eax
  int v49; // edi
  unsigned int v50; // edx
  __int64 *v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rdi
  __int64 v55; // rcx
  _QWORD *v56; // rsi
  _QWORD *m; // rax
  __int64 v58; // rax
  __int64 *v59; // rbx
  __int64 v60; // rax
  _QWORD *v61; // rax
  __int64 v62; // rax
  unsigned int v63; // [rsp+8h] [rbp-88h]
  int v64; // [rsp+8h] [rbp-88h]
  unsigned int v65; // [rsp+8h] [rbp-88h]
  unsigned int v66; // [rsp+Ch] [rbp-84h]
  int v67; // [rsp+Ch] [rbp-84h]
  unsigned int v68; // [rsp+Ch] [rbp-84h]
  int v69; // [rsp+Ch] [rbp-84h]
  __int64 v70; // [rsp+10h] [rbp-80h]
  __int64 v71; // [rsp+18h] [rbp-78h]
  __int64 v72; // [rsp+20h] [rbp-70h]
  __int64 *v73; // [rsp+28h] [rbp-68h]
  __int64 v74; // [rsp+30h] [rbp-60h]
  __int64 v75; // [rsp+38h] [rbp-58h]
  int v76; // [rsp+40h] [rbp-50h]
  int v77; // [rsp+44h] [rbp-4Ch]
  __int64 v78; // [rsp+48h] [rbp-48h]
  unsigned int v79; // [rsp+58h] [rbp-38h] BYREF
  unsigned int v80[13]; // [rsp+5Ch] [rbp-34h] BYREF

  v6 = a2;
  v78 = i;
  v79 = 1;
  if ( !i )
    goto LABEL_50;
  v8 = *(_QWORD *)(i + 120);
  if ( v8 || *(_BYTE *)(i + 28) == 17 )
  {
    a4 = *(unsigned int *)(a1 + 64);
    v9 = *(_QWORD *)(a1 + 56);
    v75 = *(_QWORD *)(a1 + 16);
    v74 = *(_QWORD *)(a1 + 24);
    v73 = *(__int64 **)(a1 + 32);
    v76 = *(_DWORD *)(a1 + 40);
    v72 = *(_QWORD *)(a1 + 48);
    i = (unsigned int)(*(_DWORD *)(a1 + 128) + 1);
    *(_DWORD *)(a1 + 128) = i;
    v10 = a4 & i;
    *(_DWORD *)(a1 + 40) = i;
    a5 = v9 + 4LL * ((unsigned int)a4 & (unsigned int)i);
    v11 = *(_DWORD *)a5;
    *(_DWORD *)a5 = i;
    if ( v11 )
    {
      do
      {
        v10 = a4 & (v10 + 1);
        i = v9 + 4LL * v10;
      }
      while ( *(_DWORD *)i );
      *(_DWORD *)i = v11;
    }
    v12 = *(_DWORD *)(a1 + 68) + 1;
    *(_DWORD *)(a1 + 68) = v12;
    if ( (unsigned int)a4 < 2 * v12 )
      sub_7702C0(a1 + 56);
    for ( *(_QWORD *)(a1 + 48) = 0; v8; v8 = *(_QWORD *)(v8 + 112) )
    {
      if ( !*(_BYTE *)(v8 + 177) )
        sub_77A250(a1, v8, &v79);
    }
    v77 = 1;
    v13 = *(_QWORD *)(v78 + 112);
    if ( !v13 )
    {
LABEL_16:
      if ( v79 )
        goto LABEL_20;
      goto LABEL_21;
    }
LABEL_8:
    v80[0] = 1;
    v71 = v6;
    while ( 2 )
    {
      v14 = v13 >> 3;
      for ( i = (unsigned int)(v13 >> 3) & *(_DWORD *)(a1 + 8); ; i = *(_DWORD *)(a1 + 8) & (unsigned int)(i + 1) )
      {
        v15 = (_QWORD *)(*(_QWORD *)a1 + 16LL * (unsigned int)i);
        if ( *v15 == v13 )
          break;
        if ( !*v15 )
          goto LABEL_52;
      }
      if ( !v15[1] )
      {
LABEL_52:
        if ( (*(_BYTE *)(v13 + 176) & 0x40) != 0 )
        {
          v32 = *(_QWORD *)(v13 + 120);
          if ( (*(_BYTE *)(v32 + 140) & 0xFB) == 8
            && (sub_8D4C10(*(_QWORD *)(v13 + 120), dword_4F077C4 != 2) & 1) != 0
            && ((*(_BYTE *)(v32 + 140) & 0xFB) != 8 || (sub_8D4C10(v32, dword_4F077C4 != 2) & 2) == 0)
            && *(_BYTE *)(v13 + 177) == 1
            && *(_QWORD *)(v13 + 184) )
          {
            while ( 1 )
            {
              v33 = *(_BYTE *)(v32 + 140);
              if ( v33 != 12 )
                break;
              v32 = *(_QWORD *)(v32 + 160);
            }
            v34 = 16;
            if ( (unsigned __int8)(v33 - 2) > 1u )
              v34 = sub_7764B0(a1, v32, v80);
            if ( !v80[0] )
              goto LABEL_87;
            if ( (unsigned __int8)(*(_BYTE *)(v32 + 140) - 8) > 3u )
            {
              v70 = 16;
              v36 = 16;
            }
            else
            {
              v35 = (unsigned int)(v34 + 7) >> 3;
              v36 = v35 + 9;
              if ( (((_BYTE)v35 + 9) & 7) != 0 )
                v36 = v35 + 17 - (((_BYTE)v35 + 9) & 7);
              v70 = v36;
            }
            if ( (v34 & 7) != 0 )
              v34 = v34 + 8 - (v34 & 7);
            v37 = v34 + v36;
            v38 = v34 + v36 + 16;
            if ( (*(_BYTE *)(a1 + 132) & 8) == 0 )
            {
              v61 = (_QWORD *)qword_4F082A0;
              if ( qword_4F082A0 )
              {
                qword_4F082A0 = *(_QWORD *)(qword_4F082A0 + 8);
              }
              else
              {
                v65 = v37;
                v69 = v34;
                v61 = (_QWORD *)sub_823970(0x10000);
                v34 = v69;
                v37 = v65;
              }
              *v61 = *(_QWORD *)(a1 + 152);
              *(_QWORD *)(a1 + 152) = v61;
              v61[1] = 0;
              v62 = *(_QWORD *)(a1 + 152);
              *(_BYTE *)(a1 + 132) |= 8u;
              *(_QWORD *)(a1 + 160) = 0;
              *(_QWORD *)(a1 + 144) = v62 + 24;
              *(_QWORD *)(a1 + 176) = 0;
              *(_DWORD *)(a1 + 168) = 0;
            }
            if ( (unsigned int)v38 > 0x400 )
            {
              v64 = v34;
              v68 = v37 + 32;
              v60 = sub_822B10(v37 + 32);
              v34 = v64;
              v39 = (char *)(v60 + 16);
              *(_QWORD *)v60 = *(_QWORD *)(a1 + 160);
              *(_DWORD *)(v60 + 8) = v68;
              *(_DWORD *)(v60 + 12) = *(_DWORD *)(a1 + 168);
              *(_QWORD *)(a1 + 160) = v60;
            }
            else
            {
              v39 = *(char **)(a1 + 144);
              v40 = v37 + 24 - (v38 & 7);
              if ( (v38 & 7) == 0 )
                v40 = v38;
              if ( 0x10000 - (*(_DWORD *)(a1 + 144) - *(_DWORD *)(a1 + 152)) < (unsigned int)v40 )
              {
                v63 = v40;
                v67 = v34;
                sub_772E70((_QWORD *)(a1 + 144));
                v39 = *(char **)(a1 + 144);
                v40 = v63;
                v34 = v67;
              }
              *(_QWORD *)(a1 + 144) = &v39[v40];
            }
            v66 = v34;
            v41 = (char *)memset(v39, 0, (unsigned int)v38) + v70;
            *(_DWORD *)((char *)v41 + v66) = 0;
            *(v41 - 1) = v32;
            if ( (unsigned __int8)(*(_BYTE *)(v32 + 140) - 9) <= 2u )
              *v41 = 0;
            if ( !v80[0] )
              goto LABEL_87;
            v42 = *(_DWORD *)(a1 + 8);
            v43 = *(_QWORD *)a1;
            v44 = v42 & v14;
            v45 = (__m128i *)(*(_QWORD *)a1 + 16LL * (v42 & (unsigned int)v14));
            if ( v45->m128i_i64[0] )
            {
              v46 = _mm_loadu_si128(v45);
              v45->m128i_i64[0] = v13;
              v45->m128i_i64[1] = (__int64)v41;
              do
              {
                v44 = v42 & (v44 + 1);
                v47 = (__m128i *)(v43 + 16LL * v44);
              }
              while ( v47->m128i_i64[0] );
              *v47 = v46;
            }
            else
            {
              v45->m128i_i64[0] = v13;
              v45->m128i_i64[1] = (__int64)v41;
            }
            v48 = *(_DWORD *)(a1 + 12) + 1;
            *(_DWORD *)(a1 + 12) = v48;
            if ( v42 < 2 * v48 )
              sub_7704A0(a1);
            v80[0] = sub_79CCD0(a1, *(_QWORD *)(v13 + 184), v41, v41, 0);
            if ( !v80[0] )
              goto LABEL_87;
          }
        }
      }
      v13 = *(_QWORD *)(v13 + 112);
      if ( v13 )
        continue;
      break;
    }
    a5 = v80[0];
    v6 = v71;
    if ( v80[0] )
      goto LABEL_16;
LABEL_87:
    v79 = 0;
    goto LABEL_21;
  }
  v77 = 0;
  v13 = *(_QWORD *)(i + 112);
  if ( v13 )
    goto LABEL_8;
LABEL_50:
  if ( !a2 )
    return v79;
  v77 = 0;
  do
  {
    v16 = sub_795660(a1, v6, i, a4, a5, a6);
    i = *(_QWORD *)(a1 + 72);
    v79 = v16;
    if ( (*(_BYTE *)(i + 48) & 0xF) != 0 )
      break;
    v6 = *(_QWORD *)(v6 + 16);
    if ( !v16 )
      break;
LABEL_20:
    ;
  }
  while ( v6 );
LABEL_21:
  if ( v77 )
  {
    for ( j = *(_QWORD *)(v78 + 120); j; j = *(_QWORD *)(j + 112) )
    {
      v18 = *(_QWORD *)(j + 120);
      for ( k = *(_BYTE *)(v18 + 140); k == 12; k = *(_BYTE *)(v18 + 140) )
        v18 = *(_QWORD *)(v18 + 160);
      if ( k == 6 )
      {
        v49 = *(_DWORD *)(a1 + 8);
        v50 = v49 & (j >> 3);
        v51 = (__int64 *)(*(_QWORD *)a1 + 16LL * v50);
        v52 = *v51;
        if ( j == *v51 )
        {
LABEL_91:
          v53 = v51[1];
          if ( v53 && v79 && (*(_BYTE *)(v53 + 8) & 4) != 0 )
          {
            v54 = *(_QWORD *)(v53 + 16);
            v55 = 2;
            v56 = *(_QWORD **)v54;
            for ( m = **(_QWORD ***)v54; m; ++v55 )
            {
              v56 = m;
              m = (_QWORD *)*m;
            }
            *v56 = qword_4F08088;
            *(_BYTE *)(v53 + 8) &= ~4u;
            v58 = *(_QWORD *)(v54 + 24);
            qword_4F08080 += v55;
            qword_4F08088 = v54;
            *(_QWORD *)(v53 + 16) = v58;
          }
        }
        else
        {
          while ( v52 )
          {
            v50 = v49 & (v50 + 1);
            v51 = (__int64 *)(*(_QWORD *)a1 + 16LL * v50);
            v52 = *v51;
            if ( *v51 == j )
              goto LABEL_91;
          }
        }
      }
      sub_77A750(a1, j);
    }
    if ( *(_QWORD *)(a1 + 48) && v79 )
      v79 = sub_799890(a1);
    v20 = *(_DWORD *)(a1 + 40);
    v21 = *(_DWORD *)(a1 + 64);
    v22 = *(_QWORD *)(a1 + 56);
    v23 = *(__int64 **)(a1 + 32);
    v24 = v21 & v20;
    for ( n = (_DWORD *)(v22 + 4LL * (v21 & v20)); v20 != *n; n = (_DWORD *)(v22 + 4LL * v24) )
      v24 = v21 & (v24 + 1);
    *n = 0;
    if ( *(_DWORD *)(v22 + 4LL * ((v24 + 1) & v21)) )
      sub_771390(*(_QWORD *)(a1 + 56), *(_DWORD *)(a1 + 64), v24);
    --*(_DWORD *)(a1 + 68);
    *(_QWORD *)(a1 + 16) = v75;
    *(_DWORD *)(a1 + 40) = v76;
    *(_QWORD *)(a1 + 24) = v74;
    *(_QWORD *)(a1 + 48) = v72;
    *(_QWORD *)(a1 + 32) = v73;
    if ( v23 != v73 && v23 )
    {
LABEL_36:
      v26 = *((unsigned int *)v23 + 3);
      v27 = *(_DWORD *)(a1 + 64);
      v28 = *(_QWORD *)(a1 + 56);
      v29 = v27 & *((_DWORD *)v23 + 3);
      v30 = *(unsigned int *)(v28 + 4LL * v29);
      if ( (_DWORD)v26 )
      {
        while ( (_DWORD)v26 != (_DWORD)v30 )
        {
          if ( !(_DWORD)v30 )
          {
            v59 = (__int64 *)*v23;
            sub_822B90(v23, *((unsigned int *)v23 + 2), v30, v26);
            if ( v59 )
            {
              v23 = v59;
              goto LABEL_36;
            }
            v23 = 0;
            break;
          }
          v29 = v27 & (v29 + 1);
          v30 = *(unsigned int *)(v28 + 4LL * v29);
        }
      }
      *(_QWORD *)(a1 + 32) = v23;
    }
  }
  return v79;
}
