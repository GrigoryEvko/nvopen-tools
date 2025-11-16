// Function: sub_6C4D80
// Address: 0x6c4d80
//
char __fastcall sub_6C4D80(_QWORD *a1, __m128i *a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v5; // r14
  int v6; // eax
  __m128i *v7; // rcx
  int v8; // r13d
  __int64 v9; // rax
  _DWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  int v23; // eax
  __int64 v24; // rdx
  __int16 v25; // r12
  int v26; // r15d
  int v27; // r12d
  __int64 v28; // r15
  int v29; // r13d
  __int64 *v30; // r12
  __int64 v31; // rax
  char i; // dl
  _DWORD *v33; // rcx
  __int64 v34; // rax
  char j; // dl
  int v36; // r15d
  int v37; // r12d
  int v38; // eax
  __int64 v39; // rax
  __int64 v40; // r15
  __int64 v41; // rsi
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rax
  _QWORD *v51; // rax
  char v52; // dl
  unsigned int v54; // [rsp+4h] [rbp-4CCh]
  int v55; // [rsp+10h] [rbp-4C0h]
  char *s; // [rsp+18h] [rbp-4B8h]
  __int64 v57; // [rsp+20h] [rbp-4B0h]
  __int64 v59; // [rsp+38h] [rbp-498h] BYREF
  _QWORD v60[2]; // [rsp+40h] [rbp-490h] BYREF
  __m128i v61; // [rsp+50h] [rbp-480h]
  __m128i v62; // [rsp+60h] [rbp-470h]
  __m128i v63; // [rsp+70h] [rbp-460h]
  _BYTE v64[352]; // [rsp+80h] [rbp-450h] BYREF
  _DWORD v65[20]; // [rsp+1E0h] [rbp-2F0h] BYREF
  __int64 v66; // [rsp+230h] [rbp-2A0h]
  _BYTE v67[400]; // [rsp+340h] [rbp-190h] BYREF

  v5 = word_4F06418[0];
  if ( word_4F06418[0] == 72 )
  {
    v54 = dword_4F06650[0];
    sub_7B8B50(a1, a2, a3, a4);
    v60[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v61 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v62 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v63 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v60[1] = *(_QWORD *)&dword_4F077C8;
    sub_878540("__cudaPushCallConfiguration", 0x1Bu);
    v39 = sub_7D5DD0(v60, 0);
    v40 = v39;
    if ( !v39 || *(_BYTE *)(v39 + 80) != 11 )
      sub_684AA0(0xBu, 0xE46u, &dword_4F063F8);
    v41 = *(_QWORD *)(v40 + 88);
    v59 = *(_QWORD *)((char *)a1 + 76);
    sub_831320(*(_QWORD *)(v41 + 152), v41, v65);
    sub_6EAB60(v40, 0, 0, (_DWORD)a1 + 68, (unsigned int)&v59, 0, (__int64)v67);
    sub_6F5960(v67, 1, 0, v42, v43, v44);
    s = (char *)sub_6BDC10(0x2Au, 0, 0, 1);
    if ( dword_4F04C44 == -1 )
    {
      v50 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v50 + 6) & 6) == 0
        && *(_BYTE *)(v50 + 4) != 12
        && (!s || !*(_QWORD *)s || (v51 = **(_QWORD ***)s) == 0 || !*v51) )
      {
        sub_684AA0(4u, 0xE47u, &dword_4F063F8);
      }
    }
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    v66 = *(_QWORD *)&dword_4F063F8;
    unk_4F061D8 = qword_4F063F0;
    sub_849040(s, v65);
    if ( word_4F06418[0] == 42 && (unsigned __int16)sub_7BE840(0, 0) == 44 )
    {
      sub_7B8B50(0, 0, v46, v47);
      sub_7B8B50(0, 0, v48, v49);
    }
    else
    {
      sub_6851C0(0xDB5u, &dword_4F063F8);
    }
    sub_7022F0(
      (unsigned int)v67,
      0,
      v65[8],
      1,
      0,
      0,
      0,
      0,
      (__int64)a1 + 68,
      (__int64)&dword_4F077C8,
      (__int64)&dword_4F077C8,
      (__int64)v64,
      0,
      0);
    sub_6E1990(s);
    if ( (unsigned int)sub_7BE5B0(27, 125, 0, 0) )
      sub_826A20(v54, dword_4F06650[0]);
    v45 = sub_8D2340(*a1);
    v7 = (__m128i *)v67;
    v8 = v45;
  }
  else
  {
    v6 = sub_8D2340(*a1);
    v7 = (__m128i *)a3;
    v8 = v6;
  }
  sub_6C0E20((__int64)a1, a2, 0, v7);
  if ( (v8 || *((_BYTE *)a1 + 16) == 1 && *(_BYTE *)(a1[18] + 56LL) == 3) && unk_4D03B90 >= 0 )
  {
    v11 = *(_QWORD *)(unk_4D03B98 + 176LL * unk_4D03B90 + 56);
    if ( v11 )
    {
      if ( !*(_BYTE *)(v11 + 40) )
      {
        v12 = *(_QWORD *)(v11 + 72);
        if ( v12 )
          *(_QWORD *)(*(_QWORD *)(a3 + 144) + 64LL) = v12;
      }
    }
  }
  v9 = sub_6EB5C0(a1);
  if ( dword_4F04C44 == -1 )
  {
    v24 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v24 + 6) & 6) == 0 && *(_BYTE *)(v24 + 4) != 12 )
    {
      if ( v9 )
      {
        if ( !*(_BYTE *)(v9 + 174) )
        {
          v25 = *(_WORD *)(v9 + 176);
          if ( (unsigned __int16)(v25 - 25761) <= 3u )
          {
            if ( unk_4D045E8 != 90 || !unk_4D045E4 )
              sub_684AA0(7u, 0xE90u, (_DWORD *)(a3 + 68));
            v57 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 144) + 72LL) + 16LL);
            if ( v25 == 25761 )
            {
              v55 = 16;
              v27 = 9;
            }
            else if ( v25 == 25762 )
            {
              v55 = 16;
              v27 = 8;
            }
            else
            {
              v26 = 32;
              if ( v25 == 25763 )
                v26 = 8;
              v55 = v26;
              v27 = 3 * (v25 != 25763) + 6;
            }
            v28 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 144) + 72LL) + 16LL);
            v29 = 0;
            do
            {
              if ( *(_BYTE *)(v28 + 24) != 2 || *(_BYTE *)(*(_QWORD *)(v28 + 56) + 173LL) != 1 )
                sub_684AA0(7u, 0xE91u, (_DWORD *)(v28 + 28));
              ++v29;
              v28 = *(_QWORD *)(v28 + 16);
            }
            while ( v27 != v29 );
            v30 = *(__int64 **)(v28 + 16);
            if ( !v30 || !v30[2] )
            {
              sub_684AA0(7u, 0xE92u, (_DWORD *)(a3 + 68));
              goto LABEL_6;
            }
            v31 = *v30;
            for ( i = *(_BYTE *)(*v30 + 140); i == 12; i = *(_BYTE *)(v31 + 140) )
              v31 = *(_QWORD *)(v31 + 160);
            if ( i == 6 )
            {
              if ( (*(_BYTE *)(v31 + 168) & 1) == 0 )
              {
                do
                {
                  v31 = *(_QWORD *)(v31 + 160);
                  v52 = *(_BYTE *)(v31 + 140);
                }
                while ( v52 == 12 );
                if ( v52 == 1 || v52 == 2 && (unsigned __int8)(*(_BYTE *)(v31 + 160) - 5) <= 1u )
                  goto LABEL_44;
              }
            }
            else if ( i == 2 && *(_BYTE *)(v31 + 160) == 10 )
            {
              goto LABEL_44;
            }
            sub_684AA0(7u, 0xE93u, (_DWORD *)v30 + 7);
LABEL_44:
            v33 = (_DWORD *)v30[2];
            v34 = *(_QWORD *)v33;
            for ( j = *(_BYTE *)(*(_QWORD *)v33 + 140LL); j == 12; j = *(_BYTE *)(v34 + 140) )
              v34 = *(_QWORD *)(v34 + 160);
            if ( j != 2 || *(_BYTE *)(v34 + 160) != 10 )
              sub_684AA0(7u, 0xE94u, v33 + 7);
            v36 = sub_620FD0(*(_QWORD *)(v57 + 56), v65);
            v37 = sub_620FD0(*(_QWORD *)(*(_QWORD *)(v57 + 16) + 56LL), v65);
            v38 = sub_620FD0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v57 + 16) + 16LL) + 56LL), v65);
            if ( (unsigned int)(v37 - 8) > 0xF8 || v36 != 64 || (v37 & 7) != 0 || v55 != v38 )
            {
              snprintf((char *)v65, 0x100u, "m%un%uk%u", v36, v37, v38);
              sub_6849F0(7u, 0xE95u, (_DWORD *)(v57 + 28), (__int64)v65);
            }
          }
        }
      }
    }
  }
LABEL_6:
  v10 = &dword_4F077C4;
  if ( dword_4F077C4 != 2
    || (LOBYTE(v10) = qword_4D03C50, qword_4D03C50) && (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x40) != 0 )
  {
    if ( v5 != 72 )
      return (char)v10;
  }
  else
  {
    v13 = sub_6EB5C0(a1);
    LOBYTE(v10) = sub_691790(v13, v5 == 72, (_DWORD *)a1 + 17);
    if ( v5 != 72 )
      return (char)v10;
  }
  v59 = *(_QWORD *)&dword_4F063F8;
  v60[0] = *(_QWORD *)&dword_4F063F8;
  sub_6E7080(v65, 0);
  v18 = sub_72CBE0(v65, 0, v14, v15, v16, v17);
  sub_6F7220(v65, v18);
  v23 = sub_72CBE0(v65, v18, v19, v20, v21, v22);
  LOBYTE(v10) = sub_6FFD30(
                  (unsigned int)v64,
                  (unsigned int)v65,
                  (unsigned int)v67,
                  v23,
                  0,
                  1,
                  0,
                  0,
                  0,
                  (__int64)&v59,
                  (__int64)v60,
                  a3);
  return (char)v10;
}
