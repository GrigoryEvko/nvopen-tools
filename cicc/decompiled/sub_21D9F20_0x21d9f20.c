// Function: sub_21D9F20
// Address: 0x21d9f20
//
__int64 __fastcall sub_21D9F20(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned int v7; // edx
  _DWORD *v8; // rax
  int v9; // ecx
  int v10; // eax
  char v11; // al
  unsigned __int8 *v12; // rdx
  __int64 i; // rax
  __int64 v14; // rax
  __int16 v15; // ax
  __int16 v16; // ax
  __int16 v17; // ax
  __int16 v18; // ax
  __int16 v19; // ax
  __int16 v20; // ax
  __int16 v21; // ax
  __int16 v22; // ax
  __int16 v23; // ax
  __int64 j; // rax
  _BYTE *v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rcx
  __int64 v28; // rax
  unsigned __int8 *v29; // rdx
  __int64 v30; // rax
  unsigned int *v31; // r13
  __int64 k; // r14
  __int64 v33; // r14
  unsigned int *v34; // rdx
  __int64 v35; // rax
  unsigned int *v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 (__fastcall *v39)(__int64); // rax
  __int64 v40; // rsi
  __int64 v41; // rcx
  _BYTE *v42; // rsi
  __int64 v43; // rcx
  int v45; // r11d
  _DWORD *v46; // r10
  int v47; // edx
  int v48; // edx
  int v49; // r8d
  int v50; // r8d
  __int64 v51; // r10
  unsigned int v52; // ecx
  int v53; // edi
  int v54; // r9d
  _DWORD *v55; // rsi
  int v56; // r8d
  int v57; // r8d
  __int64 v58; // r10
  int v59; // r9d
  unsigned int v60; // ecx
  int v61; // edi

  sub_20A1920((_QWORD *)a1, a2);
  *(_QWORD *)(a1 + 81544) = a2;
  *(_QWORD *)(a1 + 81552) = a3;
  *(_DWORD *)(a1 + 81560) = 0;
  *(_QWORD *)a1 = &unk_4A03658;
  *(_DWORD *)(a1 + 81500) = -1;
  *(_DWORD *)(a1 + 81508) = -1;
  *(_DWORD *)(a1 + 81528) = -1;
  *(_DWORD *)(a1 + 60) = 2;
  *(_QWORD *)(a1 + 64) = 0x200000002LL;
  sub_1F40C40(a1, 1);
  v5 = *(_DWORD *)(a1 + 48);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_46;
  }
  v6 = *(_QWORD *)(a1 + 32);
  v7 = ((_WORD)v5 - 1) & 0x940;
  v8 = (_DWORD *)(v6 + 8LL * (((_WORD)v5 - 1) & 0x940));
  v9 = *v8;
  if ( *v8 == 64 )
    goto LABEL_3;
  v45 = 1;
  v46 = 0;
  while ( v9 != -1 )
  {
    if ( !v46 && v9 == -2 )
      v46 = v8;
    v7 = (v5 - 1) & (v45 + v7);
    v8 = (_DWORD *)(v6 + 8LL * v7);
    v9 = *v8;
    if ( *v8 == 64 )
      goto LABEL_3;
    ++v45;
  }
  v47 = *(_DWORD *)(a1 + 40);
  if ( v46 )
    v8 = v46;
  ++*(_QWORD *)(a1 + 24);
  v48 = v47 + 1;
  if ( 4 * v48 >= 3 * v5 )
  {
LABEL_46:
    sub_1392B70(a1 + 24, 2 * v5);
    v49 = *(_DWORD *)(a1 + 48);
    if ( v49 )
    {
      v50 = v49 - 1;
      v51 = *(_QWORD *)(a1 + 32);
      v52 = v50 & 0x940;
      v48 = *(_DWORD *)(a1 + 40) + 1;
      v8 = (_DWORD *)(v51 + 8LL * (v50 & 0x940));
      v53 = *v8;
      if ( *v8 == 64 )
        goto LABEL_42;
      v54 = 1;
      v55 = 0;
      while ( v53 != -1 )
      {
        if ( !v55 && v53 == -2 )
          v55 = v8;
        v52 = v50 & (v54 + v52);
        v8 = (_DWORD *)(v51 + 8LL * v52);
        v53 = *v8;
        if ( *v8 == 64 )
          goto LABEL_42;
        ++v54;
      }
LABEL_50:
      if ( v55 )
        v8 = v55;
      goto LABEL_42;
    }
LABEL_71:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
  if ( v5 - *(_DWORD *)(a1 + 44) - v48 <= v5 >> 3 )
  {
    sub_1392B70(a1 + 24, v5);
    v56 = *(_DWORD *)(a1 + 48);
    if ( v56 )
    {
      v57 = v56 - 1;
      v58 = *(_QWORD *)(a1 + 32);
      v55 = 0;
      v59 = 1;
      v60 = v57 & 0x940;
      v48 = *(_DWORD *)(a1 + 40) + 1;
      v8 = (_DWORD *)(v58 + 8LL * (v57 & 0x940));
      v61 = *v8;
      if ( *v8 == 64 )
        goto LABEL_42;
      while ( v61 != -1 )
      {
        if ( v61 == -2 && !v55 )
          v55 = v8;
        v60 = v57 & (v59 + v60);
        v8 = (_DWORD *)(v58 + 8LL * v60);
        v61 = *v8;
        if ( *v8 == 64 )
          goto LABEL_42;
        ++v59;
      }
      goto LABEL_50;
    }
    goto LABEL_71;
  }
LABEL_42:
  *(_DWORD *)(a1 + 40) = v48;
  if ( *v8 != -1 )
    --*(_DWORD *)(a1 + 44);
  *(_QWORD *)v8 = 64;
LABEL_3:
  v8[1] = 32;
  v10 = 1 - ((byte_4FD3D80 == 0) - 1);
  *(_BYTE *)(a1 + 4640) = 0;
  *(_DWORD *)(a1 + 72) = v10;
  *(_BYTE *)(a1 + 4646) = 0;
  *(_QWORD *)(a1 + 136) = &off_4A027A0;
  *(_BYTE *)(a1 + 24802) = 4;
  *(_QWORD *)(a1 + 152) = &off_4A02720;
  *(_BYTE *)(a1 + 24806) = 2;
  *(_QWORD *)(a1 + 160) = &off_4A025A0;
  *(_QWORD *)(a1 + 168) = &off_4A024A0;
  *(_QWORD *)(a1 + 192) = &off_4A02620;
  *(_QWORD *)(a1 + 200) = &off_4A02520;
  *(_QWORD *)(a1 + 184) = &off_4A02760;
  *(_QWORD *)(a1 + 808) = &off_4A026A0;
  *(_WORD *)(a1 + 24800) = 516;
  *(_BYTE *)(a1 + 4631) = sub_21652E0((__int64)a3) ^ 1;
  v11 = sub_21652E0((__int64)a3);
  v12 = (unsigned __int8 *)&unk_435DBF8;
  *(_BYTE *)(a1 + 24833) = 2 * (v11 == 0);
  for ( i = 8; ; i = *v12 )
  {
    ++v12;
    v14 = a1 + 259 * i;
    *(_BYTE *)(v14 + 2558) = 2;
    *(_BYTE *)(v14 + 2614) = 2;
    if ( v12 == (unsigned __int8 *)&unk_435DC01 )
      break;
  }
  v15 = *(_WORD *)(a1 + 34294);
  LOBYTE(v15) = v15 & 0xF;
  *(_BYTE *)(a1 + 4124) = 0;
  *(_BYTE *)(a1 + 3865) = 0;
  *(_WORD *)(a1 + 34294) = v15 | 0x20;
  v16 = *(_WORD *)(a1 + 34524);
  *(_BYTE *)(a1 + 3606) = 0;
  LOBYTE(v16) = v16 & 0xF;
  *(_BYTE *)(a1 + 3347) = 0;
  *(_BYTE *)(a1 + 3088) = 2;
  *(_WORD *)(a1 + 34524) = v16 | 0x20;
  v17 = *(_WORD *)(a1 + 34526);
  *(_WORD *)(a1 + 3856) = 1028;
  LOBYTE(v17) = v17 & 0xF;
  *(_BYTE *)(a1 + 3858) = 4;
  *(_BYTE *)(a1 + 4115) = 4;
  *(_WORD *)(a1 + 34526) = v17 | 0x20;
  v18 = *(_WORD *)(a1 + 53080);
  *(_WORD *)(a1 + 4116) = 1028;
  LOBYTE(v18) = v18 & 0xF;
  *(_BYTE *)(a1 + 3848) = 0;
  *(_BYTE *)(a1 + 4107) = 0;
  *(_WORD *)(a1 + 53080) = v18 | 0x20;
  v19 = *(_WORD *)(a1 + 54230);
  *(_WORD *)(a1 + 3324) = 514;
  LOBYTE(v19) = v19 & 0xF;
  *(_BYTE *)(a1 + 3583) = 2;
  *(_WORD *)(a1 + 3584) = 514;
  *(_WORD *)(a1 + 3842) = 0;
  *(_BYTE *)(a1 + 3844) = 2;
  *(_BYTE *)(a1 + 4101) = 0;
  *(_WORD *)(a1 + 4102) = 512;
  *(_WORD *)(a1 + 2870) = 2;
  *(_BYTE *)(a1 + 3729) = 4;
  *(_BYTE *)(a1 + 3988) = 4;
  *(_BYTE *)(a1 + 2725) = 4;
  *(_WORD *)(a1 + 54230) = v19 | 0x20;
  v20 = *(_WORD *)(a1 + 54238);
  *(_BYTE *)(a1 + 59701) = 2;
  LOBYTE(v20) = v20 & 0xF;
  *(_WORD *)(a1 + 54238) = v20 | 0x20;
  v21 = *(_WORD *)(a1 + 53312);
  LOBYTE(v21) = v21 & 0xF;
  *(_WORD *)(a1 + 53312) = v21 | 0x20;
  v22 = *(_WORD *)(a1 + 54462);
  LOBYTE(v22) = v22 & 0xF;
  *(_WORD *)(a1 + 54462) = v22 | 0x20;
  v23 = *(_WORD *)(a1 + 54470);
  LOBYTE(v23) = v23 & 0xF;
  *(_WORD *)(a1 + 54470) = v23 | 0x20;
  *(_WORD *)(a1 + 59816) = 514;
  *(_WORD *)(a1 + 3125) = 1028;
  for ( j = 0; j != 690; j += 115 )
  {
    *(_BYTE *)(a1 + 2 * j + 32673) = 17;
    *(_BYTE *)(a1 + j + 58890) = 2;
  }
  *(_BYTE *)(a1 + 5023) = 0;
  v25 = (_BYTE *)(a1 + 6092);
  LOBYTE(v26) = 14;
  *(_BYTE *)(a1 + 4764) = 0;
  *(_BYTE *)(a1 + 4505) = 0;
  *(_BYTE *)(a1 + 2896) = 0;
  *(_DWORD *)(a1 + 3781) = 0;
  *(_DWORD *)(a1 + 4040) = 0;
  do
  {
    if ( (unsigned __int8)sub_21C9860(v26) )
    {
      v25[141] = 4;
      v25[142] = 4;
      *v25 = 4;
    }
    v26 = v27 + 1;
    v25 += 259;
  }
  while ( v26 != 110 );
  *(_BYTE *)(a1 + 3942) = 4;
  v28 = 4;
  v29 = (unsigned __int8 *)&unk_435DBF0;
  *(_BYTE *)(a1 + 4201) = 4;
  *(_BYTE *)(a1 + 3403) = 4;
  *(_DWORD *)(a1 + 2885) = (_DWORD)&unk_4020204;
  *(_BYTE *)(a1 + 4163) = 4;
  *(_BYTE *)(a1 + 3904) = 4;
  *(_BYTE *)(a1 + 3243) = 4;
  *(_BYTE *)(a1 + 4279) = 4;
  *(_BYTE *)(a1 + 13085) = 4;
  *(_BYTE *)(a1 + 13344) = 4;
  *(_BYTE *)(a1 + 13603) = 4;
  *(_BYTE *)(a1 + 13862) = 4;
  *(_BYTE *)(a1 + 14121) = 4;
  *(_BYTE *)(a1 + 14380) = 4;
  *(_BYTE *)(a1 + 14639) = 4;
  *(_BYTE *)(a1 + 14898) = 4;
  *(_BYTE *)(a1 + 4456) = 4;
  *(_BYTE *)(a1 + 4458) = 4;
  while ( 1 )
  {
    ++v29;
    v30 = a1 + 259 * v28;
    *(_BYTE *)(v30 + 2543) = 0;
    *(_DWORD *)(v30 + 2536) = 0;
    *(_WORD *)(v30 + 2551) = 0;
    if ( v29 == (unsigned __int8 *)&unk_435DBF3 )
      break;
    v28 = *v29;
  }
  *(_BYTE *)(a1 + 74029) |= 0x40u;
  *(_BYTE *)(a1 + 74024) |= 0x10u;
  *(_BYTE *)(a1 + 74021) |= 0x50u;
  *(_BYTE *)(a1 + 74030) |= 4u;
  *(_BYTE *)(a1 + 74022) |= 6u;
  *(_BYTE *)(a1 + 3586) = 2;
  *(_BYTE *)(a1 + 3845) = 2;
  *(_BYTE *)(a1 + 4104) = 2;
  *(_BYTE *)(a1 + 3074) = 4;
  *(_WORD *)(a1 + 4035) = 514;
  if ( (unsigned __int8)sub_21652E0((__int64)a3) )
    *(_BYTE *)(a1 + 74032) |= 2u;
  v31 = (unsigned int *)&unk_435DBE0;
  for ( k = 76; ; k = *v31 )
  {
    *(_BYTE *)(a1 + k + 4494) = sub_21652E0((__int64)a3) ^ 1;
    v33 = *v31++;
    *(_BYTE *)(a1 + v33 + 24696) = 2 * ((unsigned __int8)sub_21652E0((__int64)a3) == 0);
    if ( &unk_435DBF0 == (_UNKNOWN *)v31 )
      break;
  }
  v34 = (unsigned int *)&unk_435DBC0;
  *(_BYTE *)(a1 + 4656) = 2;
  v35 = 174;
  *(_BYTE *)(a1 + 24858) = 2;
  while ( 1 )
  {
    ++v34;
    *(_BYTE *)(a1 + v35 + 4494) = 0;
    *(_BYTE *)(a1 + v35 + 4753) = 0;
    *(_BYTE *)(a1 + v35 + 5012) = 0;
    *(_BYTE *)(a1 + v35 + 24696) = 2;
    if ( v34 == (unsigned int *)&unk_435DBD8 )
      break;
    v35 = *v34;
  }
  v36 = (unsigned int *)&unk_435DBA0;
  *(_BYTE *)(a1 + 4595) = 2;
  v37 = 79;
  *(_BYTE *)(a1 + 24797) = 2;
  *(_BYTE *)(a1 + 4854) = 4;
  *(_BYTE *)(a1 + 5113) = 4;
  while ( 1 )
  {
    ++v36;
    *(_BYTE *)(a1 + v37 + 4494) = 1;
    *(_BYTE *)(a1 + v37 + 4753) = 0;
    *(_BYTE *)(a1 + v37 + 5012) = 0;
    *(_BYTE *)(a1 + v37 + 24696) = 2;
    if ( v36 == (unsigned int *)&unk_435DBC0 )
      break;
    v37 = *v36;
  }
  v38 = *a3;
  *(_DWORD *)(a1 + 4674) = (_DWORD)&loc_1010101;
  *(_WORD *)(a1 + 4281) = 1028;
  v39 = *(__int64 (__fastcall **)(__int64))(v38 + 112);
  if ( v39 == sub_214AB90 )
    v40 = (__int64)(a3 + 40);
  else
    v40 = v39((__int64)a3);
  sub_1F41BE0(a1, v40);
  if ( !(unsigned int)sub_1700720(*(_QWORD *)(a1 + 81544)) )
    *(_BYTE *)(a1 + 3077) = 4;
  *(_DWORD *)(a1 + 2999) = (_DWORD)&loc_2020202;
  LOBYTE(v41) = 14;
  v42 = (_BYTE *)(a1 + 6091);
  *(_DWORD *)(a1 + 3258) = (_DWORD)&loc_2020202;
  *(_DWORD *)(a1 + 3517) = (_DWORD)&loc_2020202;
  *(_DWORD *)(a1 + 3776) = (_DWORD)&loc_2020202;
  *(_DWORD *)(a1 + 4035) = (_DWORD)&loc_2020202;
  *(_BYTE *)(a1 + 2724) = 4;
  do
  {
    if ( (unsigned __int8)sub_21C9860(v41) )
      *v42 = 4;
    v41 = v43 + 1;
    v42 += 259;
  }
  while ( v41 != 110 );
  *(_BYTE *)(a1 + 74029) |= 0x80u;
  *(_BYTE *)(a1 + 3942) = 4;
  *(_BYTE *)(a1 + 4201) = 4;
  *(_BYTE *)(a1 + 3403) = 4;
  *(_DWORD *)(a1 + 2885) = (_DWORD)&unk_4020204;
  *(_BYTE *)(a1 + 4026) = 4;
  *(_BYTE *)(a1 + 3507) = 4;
  *(_WORD *)(a1 + 3766) = 1028;
  *(_WORD *)(a1 + 2880) = 1028;
  *(_BYTE *)(a1 + 4833) = 4;
  *(_BYTE *)(a1 + 5092) = 4;
  return 1028;
}
