// Function: sub_623870
// Address: 0x623870
//
__int64 __fastcall sub_623870(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 v8; // rcx
  char v9; // di
  char v10; // r12
  bool v11; // r12
  char v12; // r13
  unsigned __int8 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int v18; // r15d
  __int16 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  char v24; // al
  __int64 v25; // rax
  __int64 result; // rax
  char v27; // r8
  unsigned int v28; // r14d
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r9
  int v33; // r11d
  __int64 v34; // r14
  __int64 v35; // r10
  int v36; // ecx
  unsigned int v37; // eax
  __int64 v38; // rsi
  __int64 *v39; // rdi
  __int64 v40; // r8
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // r10
  int v44; // r11d
  unsigned int v45; // edx
  int v46; // eax
  __int64 v47; // rcx
  _BYTE *v48; // rax
  unsigned int v49; // edx
  int v50; // eax
  unsigned int v51; // ebx
  _QWORD *v52; // rax
  unsigned int v53; // r10d
  _QWORD *v54; // rcx
  _QWORD *v55; // rsi
  __int64 *v56; // r9
  __int64 *v57; // rsi
  unsigned __int64 v58; // rdi
  unsigned __int64 j; // rdx
  unsigned int v60; // edx
  unsigned __int64 *v61; // rax
  _QWORD *v62; // rax
  _QWORD *v63; // rsi
  _QWORD *v64; // rcx
  __int64 *v65; // rcx
  unsigned __int64 v66; // rdi
  unsigned __int64 i; // rdx
  unsigned int v68; // edx
  unsigned __int64 *v69; // rax
  unsigned int v70; // [rsp+Ch] [rbp-1C4h]
  unsigned int v71; // [rsp+Ch] [rbp-1C4h]
  unsigned int v72; // [rsp+10h] [rbp-1C0h]
  unsigned int v73; // [rsp+10h] [rbp-1C0h]
  unsigned int v75; // [rsp+18h] [rbp-1B8h]
  unsigned int v76; // [rsp+18h] [rbp-1B8h]
  unsigned int v77; // [rsp+2Ch] [rbp-1A4h] BYREF
  _QWORD v78[44]; // [rsp+30h] [rbp-1A0h] BYREF
  int v79; // [rsp+190h] [rbp-40h] BYREF
  __int16 v80; // [rsp+194h] [rbp-3Ch]

  v6 = 776LL * dword_4F04C64;
  v7 = qword_4F04C68[0] + v6;
  v8 = *(unsigned __int8 *)(qword_4F04C68[0] + v6 + 4);
  if ( (_BYTE)v8 != 1 )
    goto LABEL_2;
  if ( (*(_BYTE *)(a3 + 129) & 0x40) != 0 )
  {
    if ( word_4F06418[0] != 16 )
    {
      if ( (_DWORD)a2 && (*(_BYTE *)(v7 + 6) & 0x40) == 0 )
      {
LABEL_32:
        memset(v78, 0, sizeof(v78));
        v28 = dword_4F06650[0];
        v79 = 0;
        v80 = 0;
        BYTE4(v78[3]) = 1;
        BYTE3(v78[9]) = 1;
        if ( !a1 || *(_QWORD *)(a3 + 368) && (*(_BYTE *)(a3 + 133) & 8) == 0 )
          return sub_7BDFF0(v78, 1);
        *(_BYTE *)a1 |= 0x20u;
        v29 = ((__int64 (*)(void))sub_7ADF90)();
        *(_QWORD *)(a1 + 8) = v29;
        sub_7ADF70(v29, 1);
        sub_7C6880(*(_QWORD *)(a1 + 8), v78);
        if ( dword_4F04C44 != -1
          || (v30 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v30 + 6) & 6) != 0)
          || *(_BYTE *)(v30 + 4) == 12 )
        {
          v47 = dword_4F06650[0] - 1;
          if ( (unsigned int)v47 < v28 )
            v47 = v28;
          v48 = (_BYTE *)sub_888280(0, 0, v28, v47);
          v48[51] = 1;
          v48[50] = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) == 0;
          v48[52] = sub_899810();
        }
        return sub_7AE210(*(_QWORD *)(a1 + 8));
      }
      goto LABEL_7;
    }
    goto LABEL_27;
  }
  v27 = *(_BYTE *)(qword_4F04C68[0] + 776LL * *(int *)(v7 + 344) + 4);
  if ( ((v27 - 15) & 0xFD) == 0 || v27 == 2 )
  {
    if ( word_4F06418[0] != 16 )
    {
      if ( !(_DWORD)a2 || dword_4F04C44 != -1 )
        goto LABEL_7;
      v9 = *(_BYTE *)(v7 + 6);
      goto LABEL_50;
    }
  }
  else
  {
LABEL_2:
    if ( word_4F06418[0] != 16 )
    {
      if ( !(_DWORD)a2 )
        goto LABEL_7;
      v9 = *(_BYTE *)(v7 + 6);
      a2 = 1;
      if ( dword_4F04C44 != -1 )
      {
LABEL_5:
        if ( (v9 & 0x40) != 0 || !(_BYTE)a2 )
          goto LABEL_7;
        goto LABEL_32;
      }
      a2 = 0;
LABEL_50:
      if ( (v9 & 6) != 0
        || (_BYTE)v8 == 12
        || unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
      {
        a2 = ((unsigned __int8)a2 ^ 1) & 1;
        goto LABEL_5;
      }
LABEL_7:
      v10 = *(_BYTE *)(v7 + 13);
      *(_BYTE *)(v7 + 13) = v10 | 2;
      v11 = (v10 & 2) != 0;
      v12 = dword_4F06978;
      if ( dword_4F06978 )
      {
        v13 = *(_BYTE *)(v7 + 6);
        *(_BYTE *)(v7 + 6) = v13 | 0x80;
        v12 = v13 >> 7;
      }
      sub_7296C0(&v77);
      if ( *(_BYTE *)(v7 + 4) != 1 || *(_BYTE *)(v7 - 772) != 6 || (*(_BYTE *)(a3 + 8) & 8) == 0 || *(_QWORD *)a3 )
      {
        v18 = dword_4F063F8;
        v19 = unk_4F063FC;
        v78[0] = sub_724DC0(&v77, a2, v14, v15, v16, v17);
        sub_6B9B50(v78[0]);
        if ( a1 )
        {
          v24 = *(_BYTE *)(v78[0] + 173LL);
          if ( v24 == 12 || !v24 || (unsigned int)sub_711520() )
            *(_BYTE *)a1 |= 4u;
          v25 = sub_724E50(v78, a2, v20, v21, v22, v23);
          *(_QWORD *)(a1 + 8) = v25;
          *(_DWORD *)(v25 + 64) = v18;
          *(_WORD *)(v25 + 68) = v19;
        }
        else
        {
          sub_724E30(v78);
        }
        goto LABEL_16;
      }
      memset(v78, 0, sizeof(v78));
      v79 = 0;
      v80 = 0;
      BYTE4(v78[3]) = 1;
      BYTE3(v78[9]) = 1;
      if ( !a1 )
      {
        sub_7BDFF0(v78, 1);
        goto LABEL_16;
      }
      *(_BYTE *)a1 |= 0x20u;
      v31 = sub_7ADF90(&v79, a2, 0);
      *(_QWORD *)(a1 + 8) = v31;
      sub_7ADF70(v31, 1);
      sub_7C6880(*(_QWORD *)(a1 + 8), v78);
      sub_7AE210(*(_QWORD *)(a1 + 8));
      v32 = *(_QWORD *)(v7 - 568);
      v33 = *(_DWORD *)v7;
      v34 = qword_4CFDE40;
      v35 = *a4;
      v36 = *(_DWORD *)(qword_4CFDE40 + 8);
      v37 = v36 & (a1 >> 3);
      v38 = *(_QWORD *)qword_4CFDE40 + 32LL * v37;
      if ( *(_QWORD *)v38 )
      {
        do
        {
          v37 = v36 & (v37 + 1);
          v39 = (__int64 *)(*(_QWORD *)qword_4CFDE40 + 32LL * v37);
        }
        while ( *v39 );
        sub_622D80(v39, (__int64 *)v38);
        v41 = *(_QWORD *)v34 + v40;
        *(_QWORD *)v41 = a1;
        *(_QWORD *)(v41 + 8) = v42;
        *(_QWORD *)(v41 + 16) = v43;
        *(_DWORD *)(v41 + 24) = v44;
        v45 = *(_DWORD *)(v34 + 8);
        v46 = *(_DWORD *)(v34 + 12) + 1;
        *(_DWORD *)(v34 + 12) = v46;
        if ( 2 * v46 <= v45 )
          goto LABEL_45;
        v73 = v45;
        v71 = v45 + 1;
        v51 = 2 * v45 + 1;
        v76 = 2 * v45 + 2;
        v62 = (_QWORD *)sub_823970(32LL * v76);
        v53 = v71;
        v63 = v62;
        if ( v76 )
        {
          v64 = &v62[4 * v51 + 4];
          do
          {
            if ( v62 )
              *v62 = 0;
            v62 += 4;
          }
          while ( v64 != v62 );
        }
        v56 = *(__int64 **)v34;
        if ( v71 )
        {
          v65 = *(__int64 **)v34;
          do
          {
            v66 = *v65;
            if ( *v65 )
            {
              for ( i = v66 >> 3; ; LODWORD(i) = v68 + 1 )
              {
                v68 = v51 & i;
                v69 = &v63[4 * v68];
                if ( !*v69 )
                  break;
              }
              *v69 = v66;
              *(__m128i *)(v69 + 1) = _mm_loadu_si128((const __m128i *)(v65 + 1));
              v69[3] = v65[3];
            }
            v65 += 4;
          }
          while ( &v56[4 * v73 + 4] != v65 );
        }
        *(_QWORD *)v34 = v63;
      }
      else
      {
        *(_QWORD *)v38 = a1;
        *(_QWORD *)(v38 + 8) = v32;
        *(_QWORD *)(v38 + 16) = v35;
        *(_DWORD *)(v38 + 24) = v33;
        v49 = *(_DWORD *)(v34 + 8);
        v50 = *(_DWORD *)(v34 + 12) + 1;
        *(_DWORD *)(v34 + 12) = v50;
        if ( 2 * v50 <= v49 )
        {
LABEL_45:
          sub_643E40(sub_622D60, a3, 1);
LABEL_16:
          sub_729730(v77);
          result = 776LL * dword_4F04C64;
          if ( dword_4F06978 )
            *(_BYTE *)(qword_4F04C68[0] + result + 6) = *(_BYTE *)(qword_4F04C68[0] + result + 6) & 0x7F | (v12 << 7);
          *(_BYTE *)(qword_4F04C68[0] + result + 13) = *(_BYTE *)(qword_4F04C68[0] + result + 13) & 0xFD | (2 * v11);
          return result;
        }
        v72 = v49;
        v70 = v49 + 1;
        v51 = 2 * v49 + 1;
        v75 = 2 * v49 + 2;
        v52 = (_QWORD *)sub_823970(32LL * v75);
        v53 = v70;
        v54 = v52;
        if ( v75 )
        {
          v55 = &v52[4 * v51 + 4];
          do
          {
            if ( v52 )
              *v52 = 0;
            v52 += 4;
          }
          while ( v55 != v52 );
        }
        v56 = *(__int64 **)v34;
        if ( v70 )
        {
          v57 = *(__int64 **)v34;
          do
          {
            v58 = *v57;
            if ( *v57 )
            {
              for ( j = v58 >> 3; ; LODWORD(j) = v60 + 1 )
              {
                v60 = v51 & j;
                v61 = &v54[4 * v60];
                if ( !*v61 )
                  break;
              }
              *v61 = v58;
              *(__m128i *)(v61 + 1) = _mm_loadu_si128((const __m128i *)(v57 + 1));
              v61[3] = v57[3];
            }
            v57 += 4;
          }
          while ( v57 != &v56[4 * v72 + 4] );
        }
        *(_QWORD *)v34 = v54;
      }
      *(_DWORD *)(v34 + 8) = v51;
      sub_823A00(v56, 32LL * v53);
      goto LABEL_45;
    }
  }
LABEL_27:
  result = sub_7B8B50(word_4F06418[0], a2, v6, v8);
  if ( a1 )
    *(_BYTE *)a1 |= 0x20u;
  return result;
}
