// Function: sub_7A30C0
// Address: 0x7a30c0
//
__int64 __fastcall sub_7A30C0(__int64 a1, __int64 a2, int a3, __m128i *a4, __m128i *a5)
{
  _QWORD *v5; // rbx
  __m128i *v6; // r13
  bool v7; // zf
  unsigned __int64 v8; // r12
  __int64 result; // rax
  int v10; // ecx
  unsigned int v11; // edx
  unsigned int v12; // eax
  __int64 v13; // r15
  size_t v14; // rdx
  __int64 v15; // rax
  char *v16; // rcx
  __m128i *v17; // r15
  __int64 v18; // rax
  char v19; // dl
  char v20; // al
  char v21; // al
  __int64 v22; // rax
  __int64 v23; // rcx
  int v24; // r8d
  size_t v25; // rdx
  unsigned int v26; // r9d
  char *v27; // rdi
  char *v28; // rax
  unsigned int v29; // r8d
  __m128i *v30; // r9
  unsigned int v31; // eax
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rax
  int v36; // ecx
  int v37; // eax
  unsigned int v38; // edx
  unsigned int v39; // esi
  char *v40; // r15
  unsigned int v41; // eax
  __int64 v42; // rax
  __int64 v43; // r15
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-138h]
  __int64 v46; // [rsp+8h] [rbp-138h]
  size_t v47; // [rsp+10h] [rbp-130h]
  unsigned int v48; // [rsp+10h] [rbp-130h]
  int v49; // [rsp+10h] [rbp-130h]
  int v50; // [rsp+10h] [rbp-130h]
  int v51; // [rsp+18h] [rbp-128h]
  size_t v52; // [rsp+18h] [rbp-128h]
  __int64 v53; // [rsp+18h] [rbp-128h]
  size_t v54; // [rsp+18h] [rbp-128h]
  size_t v55; // [rsp+18h] [rbp-128h]
  unsigned int v56; // [rsp+18h] [rbp-128h]
  unsigned int v57; // [rsp+18h] [rbp-128h]
  unsigned int v59; // [rsp+20h] [rbp-120h]
  __m128i *v60; // [rsp+20h] [rbp-120h]
  unsigned int v61; // [rsp+20h] [rbp-120h]
  __int64 v62; // [rsp+20h] [rbp-120h]
  unsigned int v63; // [rsp+20h] [rbp-120h]
  int v66; // [rsp+3Ch] [rbp-104h]
  unsigned int v67; // [rsp+4Ch] [rbp-F4h] BYREF
  _BYTE v68[16]; // [rsp+50h] [rbp-F0h] BYREF
  void *s; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v70; // [rsp+68h] [rbp-D8h]
  __int64 v71; // [rsp+70h] [rbp-D0h]
  int v72; // [rsp+78h] [rbp-C8h]
  __int64 v73; // [rsp+80h] [rbp-C0h]
  __m128i v74; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v75; // [rsp+C0h] [rbp-80h]
  char v76; // [rsp+D4h] [rbp-6Ch]
  char v77; // [rsp+D5h] [rbp-6Bh]
  int v78; // [rsp+D8h] [rbp-68h]
  __int64 v79; // [rsp+108h] [rbp-38h]

  v5 = (_QWORD *)a1;
  v6 = *(__m128i **)a1;
  v66 = a2;
  v7 = *(_BYTE *)(*(_QWORD *)a1 + 140LL) == 12;
  v8 = *(_QWORD *)a1;
  v67 = 1;
  if ( v7 )
  {
    do
      v8 = *(_QWORD *)(v8 + 160);
    while ( *(_BYTE *)(v8 + 140) == 12 );
  }
  if ( *(_BYTE *)(a1 + 24) == 2 )
  {
    a1 = *(_QWORD *)(a1 + 56);
    v20 = *(_BYTE *)(a1 + 173);
    if ( v20 != 12
      && (v20 != 7 || (*(_BYTE *)(a1 + 192) & 2) == 0 || (*(_BYTE *)(*(_QWORD *)(a1 + 200) + 193LL) & 4) == 0) )
    {
      sub_740190(a1, a4, 0x800u);
      return v67;
    }
  }
  result = dword_4F07588;
  if ( dword_4F07588 )
  {
    result = 0;
    if ( !dword_4D03F94 )
    {
      if ( dword_4F08058 )
      {
        sub_771BE0(a1, a2);
        dword_4F08058 = 0;
      }
      sub_774A30((__int64)v68, a2);
      if ( (_DWORD)qword_4F077B4 && qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 18LL) & 2) != 0 )
        v77 |= 2u;
      v10 = 32;
      v75 = *(_QWORD *)((char *)v5 + 28);
      if ( (*((_BYTE *)v5 + 25) & 3) == 0 )
      {
        v10 = 16;
        if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 2) > 1u )
          v10 = sub_7764B0((__int64)v68, v8, &v67);
      }
      if ( !v67 )
      {
        if ( (v76 & 0x40) != 0 )
        {
          sub_72C970((__int64)a4);
          v67 = 1;
        }
        goto LABEL_37;
      }
      if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 8) > 3u )
      {
        v14 = 8;
        v13 = 16;
        v12 = 16;
      }
      else
      {
        v11 = (unsigned int)(v10 + 7) >> 3;
        v12 = v11 + 9;
        if ( (((_BYTE)v11 + 9) & 7) != 0 )
          v12 = v11 + 17 - (((_BYTE)v11 + 9) & 7);
        v13 = v12;
        v14 = v12 - 8LL;
      }
      v15 = v10 + v12;
      if ( (unsigned int)v15 > 0x400 )
      {
        v47 = v14;
        v51 = v15 + 16;
        v22 = sub_822B10((unsigned int)(v15 + 16));
        v14 = v47;
        *(_QWORD *)v22 = v71;
        *(_DWORD *)(v22 + 8) = v51;
        *(_DWORD *)(v22 + 12) = v72;
        v16 = (char *)(v22 + 16);
        v71 = v22;
      }
      else
      {
        if ( (v15 & 7) != 0 )
          v15 = (_DWORD)v15 + 8 - (unsigned int)(v15 & 7);
        v16 = (char *)s;
        if ( 0x10000 - ((int)s - (int)v70) < (unsigned int)v15 )
        {
          v48 = v15;
          v52 = v14;
          sub_772E70(&s);
          v16 = (char *)s;
          v15 = v48;
          v14 = v52;
        }
        s = &v16[v15];
      }
      v17 = (__m128i *)((char *)memset(v16, 0, v14) + v13);
      v17[-1].m128i_i64[1] = v8;
      if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 9) <= 2u )
        v17->m128i_i64[0] = 0;
      if ( !(unsigned int)sub_786210((__int64)v68, (_QWORD **)v5, (unsigned __int64)v17, v17->m128i_i8) )
      {
        if ( (v76 & 0x40) != 0 )
        {
          sub_72C970((__int64)a4);
        }
        else
        {
          v67 = 0;
          v18 = qword_4D03C50;
          if ( qword_4D03C50 )
          {
            v19 = v78;
            if ( (v78 & 1) != 0 )
              *(_BYTE *)(qword_4D03C50 + 24LL) |= 1u;
            if ( (v19 & 2) != 0 )
              *(_BYTE *)(v18 + 24) |= 2u;
          }
        }
        goto LABEL_37;
      }
      v21 = *((_BYTE *)v5 + 25);
      if ( (v21 & 3) == 0 )
        goto LABEL_54;
      if ( !a3 )
      {
        if ( (v21 & 2) != 0 )
          v6 = (__m128i *)sub_72D6A0(v6);
        else
          v6 = (__m128i *)sub_72D600(v6);
LABEL_54:
        if ( !v67 )
          goto LABEL_37;
        if ( sub_77D750((__int64)v68, v17, (__int64)v17, (__int64)v6, (__int64)a4)
          && (!v73 || (*((_BYTE *)v5 + 24) == 10 || v66) && (unsigned int)sub_799890((__int64)v68)) )
        {
          if ( !v79 )
          {
            if ( !v5[2] && (!qword_4D03C50 || *(_BYTE *)(qword_4D03C50 + 16LL)) )
              a4[9].m128i_i64[0] = (__int64)v5;
            if ( (*((_BYTE *)v5 + 26) & 1) != 0 )
              a4[10].m128i_i8[8] |= 0x20u;
            goto LABEL_37;
          }
          sub_773640((__int64)v68);
        }
LABEL_51:
        v67 = 0;
LABEL_37:
        *a5 = _mm_loadu_si128(&v74);
        sub_771990((__int64)v68);
        return v67;
      }
      if ( (v17->m128i_i8[8] & 1) != 0 )
      {
        if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 9) > 2u
          || (*(_BYTE *)(v8 + 179) & 1) == 0
          || !(unsigned int)sub_8D5070(v8) )
        {
          if ( (v76 & 0x20) == 0 )
          {
            sub_6855B0(0xA8Du, (FILE *)((char *)v5 + 28), &v74);
            sub_770D30((__int64)v68);
          }
          goto LABEL_51;
        }
        v36 = 16;
        if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 2) <= 1u
          || (v36 = sub_7764B0((__int64)v68, v8, &v67), (unsigned __int8)(*(_BYTE *)(v8 + 140) - 8) > 3u) )
        {
          v37 = 1;
          v38 = 9;
        }
        else
        {
          v38 = ((unsigned int)(v36 + 7) >> 3) + 9;
          v37 = ((unsigned __int8)((unsigned int)(v36 + 7) >> 3) + 9) & 7;
          if ( (((unsigned __int8)((unsigned int)(v36 + 7) >> 3) + 9) & 7) == 0 )
            goto LABEL_92;
        }
        v38 = v38 + 8 - v37;
LABEL_92:
        v39 = v38 + v36;
        if ( v38 + v36 > 0x400 )
        {
          v56 = v38;
          v42 = sub_822B10(v39 + 16);
          v38 = v56;
          v43 = v42;
          v44 = v71;
          *(_DWORD *)(v43 + 8) = v39 + 16;
          *(_QWORD *)v43 = v44;
          *(_DWORD *)(v43 + 12) = v72;
          v71 = v43;
          v40 = (char *)(v43 + 16);
        }
        else
        {
          if ( (v39 & 7) != 0 )
            v39 = v39 + 8 - (v39 & 7);
          v40 = (char *)s;
          if ( 0x10000 - ((int)s - (int)v70) < v39 )
          {
            v57 = v38;
            sub_772E70(&s);
            v40 = (char *)s;
            v38 = v57;
          }
          s = &v40[v39];
        }
        v62 = v38;
        memset(v40, 0, v38 - 8LL);
        v17 = (__m128i *)&v40[v62];
        v17[-1].m128i_i64[1] = v8;
        if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 9) <= 2u )
          v17->m128i_i64[0] = 0;
        sub_7790A0((__int64)v68, v17, v8, (__int64)v17);
        goto LABEL_54;
      }
      if ( (*(_BYTE *)(*v5 + 140LL) & 0xFB) == 8 && (sub_8D4C10(*v5, dword_4F077C4 != 2) & 2) != 0 )
      {
        v67 = 0;
        if ( (v76 & 0x20) != 0 )
          goto LABEL_37;
        sub_6855B0(0xAC0u, (FILE *)((char *)v5 + 28), &v74);
        sub_770D30((__int64)v68);
        goto LABEL_54;
      }
      if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 2) > 1u )
      {
        v24 = sub_7764B0((__int64)v68, v8, &v67);
        if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 8) > 3u )
        {
          v26 = v24 + 16;
          v25 = 8;
          v23 = 16;
        }
        else
        {
          v31 = (unsigned int)(v24 + 7) >> 3;
          v32 = v31 + 9;
          if ( (((_BYTE)v31 + 9) & 7) != 0 )
          {
            v41 = v31 + 17 - (((_BYTE)v31 + 9) & 7);
            v23 = v41;
            v26 = v24 + v41;
            v25 = v41 - 8LL;
          }
          else
          {
            v23 = v32;
            v26 = v24 + v32;
            v25 = v32 - 8LL;
          }
        }
        if ( v26 > 0x400 )
        {
          v45 = v23;
          v49 = v24;
          v54 = v25;
          v61 = v26 + 16;
          v33 = sub_822B10(v26 + 16);
          v25 = v54;
          v34 = v33;
          v35 = v71;
          v24 = v49;
          *(_DWORD *)(v34 + 8) = v61;
          v23 = v45;
          *(_QWORD *)v34 = v35;
          *(_DWORD *)(v34 + 12) = v72;
          v71 = v34;
          v27 = (char *)(v34 + 16);
LABEL_71:
          v53 = v23;
          v59 = v24;
          v28 = (char *)memset(v27, 0, v25);
          v29 = v59;
          v30 = (__m128i *)&v28[v53];
          *(_QWORD *)&v28[v53 - 8] = v8;
          if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 9) <= 2u )
            v30->m128i_i64[0] = 0;
          v60 = (__m128i *)&v28[v53];
          v67 = sub_7A0070((__int64)v68, (__int64)v5, v8, (__int64)v17, v29, v30, v30->m128i_i8);
          v6 = sub_73D720(v6);
          v17 = v60;
          goto LABEL_54;
        }
        if ( (v26 & 7) != 0 )
          v26 = v26 + 8 - (v26 & 7);
      }
      else
      {
        v23 = 16;
        v24 = 16;
        v25 = 8;
        v26 = 32;
      }
      v27 = (char *)s;
      if ( 0x10000 - ((int)s - (int)v70) < v26 )
      {
        v46 = v23;
        v50 = v24;
        v55 = v25;
        v63 = v26;
        sub_772E70(&s);
        v27 = (char *)s;
        v23 = v46;
        v24 = v50;
        v25 = v55;
        v26 = v63;
      }
      s = &v27[v26];
      goto LABEL_71;
    }
  }
  return result;
}
