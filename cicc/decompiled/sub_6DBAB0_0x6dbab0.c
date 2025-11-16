// Function: sub_6DBAB0
// Address: 0x6dbab0
//
__int64 __fastcall sub_6DBAB0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r15d
  _QWORD *v5; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  _QWORD *v11; // r8
  __int64 v12; // rbx
  int v13; // eax
  __int64 v14; // rsi
  __m128i *v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  int v19; // eax
  __int64 result; // rax
  __int64 v21; // rax
  __int64 i; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  bool v27; // zf
  __int64 v28; // rbx
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rdx
  unsigned int v32; // r9d
  char v33; // al
  __int64 v34; // rax
  __m128i v35; // xmm1
  __m128i v36; // xmm2
  __m128i v37; // xmm3
  __m128i v38; // xmm4
  __m128i v39; // xmm5
  __m128i v40; // xmm6
  __m128i v41; // xmm7
  __m128i v42; // xmm0
  __m128i v43; // xmm2
  __m128i v44; // xmm3
  __m128i v45; // xmm4
  __m128i v46; // xmm5
  __m128i v47; // xmm6
  __m128i v48; // xmm7
  __m128i v49; // xmm1
  __m128i v50; // xmm2
  __m128i v51; // xmm3
  __m128i v52; // xmm4
  __m128i v53; // xmm5
  __m128i v54; // xmm6
  __int64 v55; // rax
  __m128i *v56; // [rsp+18h] [rbp-5A8h]
  int v57; // [rsp+20h] [rbp-5A0h]
  __int64 v58; // [rsp+20h] [rbp-5A0h]
  __int64 v59; // [rsp+20h] [rbp-5A0h]
  __int16 v60; // [rsp+2Eh] [rbp-592h]
  __int64 v61; // [rsp+30h] [rbp-590h] BYREF
  __int64 v62; // [rsp+38h] [rbp-588h] BYREF
  __int64 v63; // [rsp+40h] [rbp-580h] BYREF
  __int64 v64; // [rsp+48h] [rbp-578h] BYREF
  _BYTE v65[160]; // [rsp+50h] [rbp-570h] BYREF
  __m128i v66[9]; // [rsp+F0h] [rbp-4D0h] BYREF
  __m128i v67; // [rsp+180h] [rbp-440h]
  __m128i v68; // [rsp+190h] [rbp-430h]
  __m128i v69; // [rsp+1A0h] [rbp-420h]
  __m128i v70; // [rsp+1B0h] [rbp-410h]
  __m128i v71; // [rsp+1C0h] [rbp-400h]
  __m128i v72; // [rsp+1D0h] [rbp-3F0h]
  __m128i v73; // [rsp+1E0h] [rbp-3E0h]
  __m128i v74; // [rsp+1F0h] [rbp-3D0h]
  __m128i v75; // [rsp+200h] [rbp-3C0h]
  __m128i v76; // [rsp+210h] [rbp-3B0h]
  __m128i v77; // [rsp+220h] [rbp-3A0h]
  __m128i v78; // [rsp+230h] [rbp-390h]
  __m128i v79; // [rsp+240h] [rbp-380h]
  __m128i v80; // [rsp+250h] [rbp-370h] BYREF
  __m128i v81; // [rsp+260h] [rbp-360h] BYREF
  __m128i v82; // [rsp+270h] [rbp-350h] BYREF
  __m128i v83; // [rsp+280h] [rbp-340h] BYREF
  __m128i v84; // [rsp+290h] [rbp-330h] BYREF
  __m128i v85; // [rsp+2A0h] [rbp-320h] BYREF
  __m128i v86; // [rsp+2B0h] [rbp-310h] BYREF
  __m128i v87; // [rsp+2C0h] [rbp-300h] BYREF
  __m128i v88; // [rsp+2D0h] [rbp-2F0h] BYREF
  __m128i v89; // [rsp+2E0h] [rbp-2E0h] BYREF
  __m128i v90; // [rsp+2F0h] [rbp-2D0h] BYREF
  __m128i v91; // [rsp+300h] [rbp-2C0h] BYREF
  __m128i v92; // [rsp+310h] [rbp-2B0h] BYREF
  __m128i v93; // [rsp+320h] [rbp-2A0h] BYREF
  __m128i v94; // [rsp+330h] [rbp-290h] BYREF
  __m128i v95; // [rsp+340h] [rbp-280h] BYREF
  __m128i v96; // [rsp+350h] [rbp-270h] BYREF
  __m128i v97; // [rsp+360h] [rbp-260h] BYREF
  __m128i v98; // [rsp+370h] [rbp-250h] BYREF
  __m128i v99; // [rsp+380h] [rbp-240h] BYREF
  __m128i v100; // [rsp+390h] [rbp-230h] BYREF
  __m128i v101; // [rsp+3A0h] [rbp-220h] BYREF
  _QWORD v102[66]; // [rsp+3B0h] [rbp-210h] BYREF

  v5 = (_QWORD *)a1;
  if ( a1 )
  {
    v7 = *(_QWORD *)a1;
    v8 = sub_6E3DA0(*(_QWORD *)a1, 0);
    v4 = *(_DWORD *)(v7 + 44);
    v9 = (__int64)&v61;
    v64 = *(_QWORD *)(v8 + 68);
    v60 = *(_WORD *)(v7 + 48);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(v7 + 64);
    sub_6E44B0(a1, &v61, &v63);
    v12 = v61;
    v56 = *(__m128i **)(*(_QWORD *)(a1 + 16) + 16LL);
  }
  else
  {
    v64 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(0, a2, a3, a4);
    v9 = 125;
    sub_7BE280(27, 125, 0, 0);
    v31 = *(_QWORD *)&dword_4F063F8;
    v63 = *(_QWORD *)&dword_4F063F8;
    v32 = dword_4F077BC;
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    memset(v102, 0, 0x1D8u);
    v102[3] = v31;
    v102[19] = v102;
    if ( v32 && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v102[22]) |= 1u;
    sub_65C7C0((__int64)v102);
    v11 = v102;
    if ( !dword_4F077BC || (a1 = (unsigned int)qword_4F077B4, (_DWORD)qword_4F077B4) )
    {
      a1 = (unsigned __int64)v102;
      sub_64EC60((__int64)v102);
    }
    v12 = v102[36];
    v56 = 0;
    v61 = v102[36];
  }
  while ( 1 )
  {
    v13 = *(unsigned __int8 *)(v12 + 140);
    if ( (_BYTE)v13 != 12 )
      break;
    v12 = *(_QWORD *)(v12 + 160);
  }
  if ( (unsigned __int8)(v13 - 9) <= 2u )
  {
    if ( dword_4F077C4 != 2 )
      goto LABEL_7;
    if ( (unsigned int)sub_8D23B0(v12) )
      sub_8AE000(v12);
    if ( dword_4F077C4 != 2 )
      goto LABEL_7;
    v33 = *(_BYTE *)(v12 + 140);
    if ( unk_4F07778 > 201102 || dword_4F07774 )
    {
      v27 = v33 == 12;
      v34 = v12;
      if ( v27 )
      {
        do
          v34 = *(_QWORD *)(v34 + 160);
        while ( *(_BYTE *)(v34 + 140) == 12 );
      }
      if ( *(char *)(*(_QWORD *)(*(_QWORD *)v34 + 96LL) + 181LL) < 0 )
        goto LABEL_7;
    }
    else
    {
      v27 = v33 == 12;
      v55 = v12;
      if ( v27 )
      {
        do
          v55 = *(_QWORD *)(v55 + 160);
        while ( *(_BYTE *)(v55 + 140) == 12 );
      }
      if ( *(char *)(*(_QWORD *)(*(_QWORD *)v55 + 96LL) + 178LL) < 0 )
        goto LABEL_7;
    }
    if ( (*(_BYTE *)(v12 + 177) & 0x20) == 0 && (unsigned int)sub_6E53E0(5, 1427, &v63) )
      sub_684B30(0x593u, &v63);
LABEL_7:
    v57 = 1;
    goto LABEL_8;
  }
  if ( (_BYTE)v13 == 14 )
    goto LABEL_7;
  v57 = sub_6E5430(a1, v9, v10, (unsigned int)(v13 - 9), v11);
  if ( v57 )
  {
    sub_6851C0(0x58Fu, &v63);
    v57 = 0;
  }
LABEL_8:
  v14 = (__int64)v65;
  sub_6E2140(4, v65, 0, 1, v5);
  *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x20u;
  if ( v5 )
  {
    v14 = (__int64)v5;
    v15 = v56;
    sub_6F85E0(v56, v5, 128, &v80, 0);
  }
  else
  {
    if ( word_4F06418[0] == 67 )
    {
      sub_6EA0A0(v61, &v80);
      word_4F06418[0] = 29;
      LOWORD(v16) = 29;
      do
      {
        v35 = _mm_loadu_si128(&v81);
        v36 = _mm_loadu_si128(&v82);
        v37 = _mm_loadu_si128(&v83);
        v38 = _mm_loadu_si128(&v84);
        v39 = _mm_loadu_si128(&v85);
        v66[0] = _mm_loadu_si128(&v80);
        v40 = _mm_loadu_si128(&v86);
        v41 = _mm_loadu_si128(&v87);
        v66[1] = v35;
        v42 = _mm_loadu_si128(&v88);
        v66[2] = v36;
        v66[3] = v37;
        v66[4] = v38;
        v66[5] = v39;
        v66[6] = v40;
        v66[7] = v41;
        v66[8] = v42;
        if ( v81.m128i_i8[0] == 2 )
        {
          v43 = _mm_loadu_si128(&v90);
          v44 = _mm_loadu_si128(&v91);
          v45 = _mm_loadu_si128(&v92);
          v46 = _mm_loadu_si128(&v93);
          v47 = _mm_loadu_si128(&v94);
          v67 = _mm_loadu_si128(&v89);
          v48 = _mm_loadu_si128(&v95);
          v49 = _mm_loadu_si128(&v96);
          v68 = v43;
          v69 = v44;
          v50 = _mm_loadu_si128(&v97);
          v51 = _mm_loadu_si128(&v98);
          v70 = v45;
          v52 = _mm_loadu_si128(&v99);
          v71 = v46;
          v53 = _mm_loadu_si128(&v100);
          v72 = v47;
          v54 = _mm_loadu_si128(&v101);
          v73 = v48;
          v74 = v49;
          v75 = v50;
          v76 = v51;
          v77 = v52;
          v78 = v53;
          v79 = v54;
        }
        else if ( v81.m128i_i8[0] == 5 || v81.m128i_i8[0] == 1 )
        {
          v67.m128i_i64[0] = v89.m128i_i64[0];
        }
        if ( (_WORD)v16 == 29 )
        {
          v14 = 0;
          v15 = v66;
          sub_6D7FC0(v66, 0, 0, 1, &v80, 0);
        }
        else
        {
          v14 = 1;
          v15 = v66;
          sub_6D30E0((__int64)v66, (__int64 *)1, 0, (__int64)&v80);
        }
        v16 = word_4F06418[0];
      }
      while ( (word_4F06418[0] & 0xFFFB) == 0x19 );
    }
    else
    {
      sub_6E5F20(253);
      v15 = &v80;
      sub_6E6260(&v80);
    }
    v4 = qword_4F063F0;
    v60 = WORD2(qword_4F063F0);
  }
  if ( !v57 || !v81.m128i_i8[0] )
    goto LABEL_12;
  v21 = v80.m128i_i64[0];
  for ( i = *(unsigned __int8 *)(v80.m128i_i64[0] + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v21 + 140) )
    v21 = *(_QWORD *)(v21 + 160);
  if ( (_BYTE)i )
  {
    v62 = sub_724DC0(v15, v14, i, v16, v17, v18);
    v58 = sub_726700(22);
    *(_QWORD *)v58 = sub_72CBE0(22, v14, v23, v24, v25, v26);
    v27 = *(_BYTE *)(v58 + 24) == 22;
    *(_QWORD *)(v58 + 56) = v61;
    if ( v27 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
    {
      sub_6E70E0(v58, v102);
      HIDWORD(v102[8]) = v63;
      LOWORD(v102[9]) = WORD2(v63);
      *(_QWORD *)dword_4F07508 = *(_QWORD *)((char *)&v102[8] + 4);
      *(_QWORD *)&dword_4F061D8 = *(_QWORD *)&dword_4F077C8;
      *(_QWORD *)((char *)&v102[9] + 4) = *(_QWORD *)&dword_4F077C8;
      sub_6E3280(v102, &dword_4F077C8);
      sub_6F6F40(v102, 0);
    }
    v28 = v58;
    *(_QWORD *)(v58 + 16) = sub_6F6F40(&v80, 0);
    v59 = sub_726700(23);
    *(_QWORD *)v59 = sub_72BA30(unk_4F06A51);
    *(_QWORD *)(v59 + 64) = v28;
    v29 = v62;
    v30 = qword_4D03C50;
    *(_BYTE *)(v59 + 56) = 0;
    sub_7197C0(v59, v29, *(_BYTE *)(v30 + 16) != 0, &v64, v102);
    if ( LODWORD(v102[0]) )
    {
      sub_6E70E0(v59, a2);
    }
    else
    {
      sub_6E6A50(v62, a2);
      *(_QWORD *)a2 = *(_QWORD *)(a2 + 272);
    }
    sub_724E30(&v62);
  }
  else
  {
LABEL_12:
    sub_6E6260(a2);
    sub_6E6450(&v80);
  }
  v19 = v64;
  *(_DWORD *)(a2 + 76) = v4;
  *(_DWORD *)(a2 + 68) = v19;
  *(_WORD *)(a2 + 72) = WORD2(v64);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  *(_WORD *)(a2 + 80) = v60;
  *(_QWORD *)&dword_4F061D8 = *(_QWORD *)(a2 + 76);
  sub_6E3280(a2, &v64);
  sub_6E3BA0(a2, &v64, 0, &v63);
  result = sub_6E2B30(a2, &v64);
  if ( !v5 )
  {
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    return sub_7BE280(28, 18, 0, 0);
  }
  return result;
}
