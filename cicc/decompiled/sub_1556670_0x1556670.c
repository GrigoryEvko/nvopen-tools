// Function: sub_1556670
// Address: 0x1556670
//
__int64 __fastcall sub_1556670(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6, char a7)
{
  __int64 result; // rax
  __int64 v9; // rax
  _QWORD *v10; // r15
  _QWORD *v11; // r14
  __int64 *v12; // rdi
  __int64 (__fastcall *v13)(__int64); // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned int v17; // esi
  __int64 v18; // r8
  int v19; // r11d
  _QWORD *v20; // r10
  unsigned int v21; // edx
  _QWORD *v22; // rdi
  __int64 v23; // rcx
  _QWORD *v24; // r15
  _QWORD *v25; // r14
  __int64 *v26; // rdi
  __int64 (__fastcall *v27)(__int64); // rax
  __int64 v28; // rdi
  int v29; // ecx
  int v30; // edx
  _BYTE *v31; // rsi
  int v32; // esi
  int v33; // esi
  __int64 v34; // r8
  unsigned int v35; // ecx
  __int64 v36; // rdi
  int v37; // r11d
  _QWORD *v38; // r9
  int v39; // esi
  int v40; // esi
  int v41; // r11d
  __int64 v42; // r8
  unsigned int v43; // ecx
  __int64 v44; // rdi
  __int64 v46; // [rsp+8h] [rbp-88h]
  __int64 v47; // [rsp+10h] [rbp-80h]
  __int64 v48; // [rsp+18h] [rbp-78h]
  __int64 v49; // [rsp+20h] [rbp-70h] BYREF
  __int64 v50; // [rsp+28h] [rbp-68h]
  __int64 v51; // [rsp+30h] [rbp-60h]
  __int64 v52; // [rsp+38h] [rbp-58h]
  _QWORD v53[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 (__fastcall *v54)(__int64 *); // [rsp+50h] [rbp-40h]
  __int64 v55; // [rsp+58h] [rbp-38h]

  *(_QWORD *)(a1 + 32) = a3;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = a4;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = a5;
  v46 = a1 + 240;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_DWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_BYTE *)(a1 + 296) = a6;
  *(_QWORD *)(a1 + 304) = 0;
  *(_BYTE *)(a1 + 297) = a7;
  *(_QWORD *)(a1 + 328) = a1 + 344;
  result = 0x800000000LL;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 336) = 0x800000000LL;
  *(_QWORD *)(a1 + 472) = a1 + 488;
  *(_QWORD *)(a1 + 480) = 0x800000000LL;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_DWORD *)(a1 + 648) = 0;
  if ( !a4 )
    return result;
  v9 = *(_QWORD *)(a4 + 32);
  v47 = a4 + 8;
  v48 = a4 + 24;
  v49 = *(_QWORD *)(a4 + 16);
  v50 = a4 + 8;
  v51 = v9;
  v52 = a4 + 24;
  if ( a4 + 24 == v9 )
    goto LABEL_17;
  while ( 1 )
  {
    do
    {
      v10 = v53;
      v55 = 0;
      v11 = v53;
      v12 = &v49;
      v54 = sub_1548AB0;
      v13 = sub_1548A90;
      if ( ((unsigned __int8)sub_1548A90 & 1) == 0 )
        goto LABEL_5;
      while ( 1 )
      {
        v13 = *(__int64 (__fastcall **)(__int64))((char *)v13 + *v12 - 1);
LABEL_5:
        v14 = v13((__int64)v12);
        if ( v14 )
          break;
        while ( 1 )
        {
          v15 = v11[3];
          v13 = (__int64 (__fastcall *)(__int64))v11[2];
          v10 += 2;
          v11 = v10;
          v12 = (__int64 *)((char *)&v49 + v15);
          if ( ((unsigned __int8)v13 & 1) != 0 )
            break;
          v14 = v13((__int64)v12);
          if ( v14 )
            goto LABEL_8;
        }
      }
LABEL_8:
      v16 = *(_QWORD *)(v14 + 48);
      v53[0] = v16;
      if ( !v16 )
        goto LABEL_11;
      v17 = *(_DWORD *)(a1 + 264);
      if ( v17 )
      {
        v18 = *(_QWORD *)(a1 + 248);
        v19 = 1;
        v20 = 0;
        v21 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v22 = (_QWORD *)(v18 + 8LL * v21);
        v23 = *v22;
        if ( v16 == *v22 )
          goto LABEL_11;
        while ( v23 != -8 )
        {
          if ( v23 != -16 || v20 )
            v22 = v20;
          v21 = (v17 - 1) & (v19 + v21);
          v23 = *(_QWORD *)(v18 + 8LL * v21);
          if ( v16 == v23 )
            goto LABEL_11;
          ++v19;
          v20 = v22;
          v22 = (_QWORD *)(v18 + 8LL * v21);
        }
        v29 = *(_DWORD *)(a1 + 256);
        if ( !v20 )
          v20 = v22;
        ++*(_QWORD *)(a1 + 240);
        v30 = v29 + 1;
        if ( 4 * (v29 + 1) < 3 * v17 )
        {
          if ( v17 - *(_DWORD *)(a1 + 260) - v30 > v17 >> 3 )
            goto LABEL_31;
          sub_15564C0(v46, v17);
          v39 = *(_DWORD *)(a1 + 264);
          if ( !v39 )
          {
LABEL_63:
            ++*(_DWORD *)(a1 + 256);
            BUG();
          }
          v16 = v53[0];
          v40 = v39 - 1;
          v41 = 1;
          v38 = 0;
          v42 = *(_QWORD *)(a1 + 248);
          v43 = v40 & ((LODWORD(v53[0]) >> 9) ^ (LODWORD(v53[0]) >> 4));
          v20 = (_QWORD *)(v42 + 8LL * v43);
          v44 = *v20;
          v30 = *(_DWORD *)(a1 + 256) + 1;
          if ( *v20 == v53[0] )
            goto LABEL_31;
          while ( v44 != -8 )
          {
            if ( v44 == -16 && !v38 )
              v38 = v20;
            v43 = v40 & (v41 + v43);
            v20 = (_QWORD *)(v42 + 8LL * v43);
            v44 = *v20;
            if ( v53[0] == *v20 )
              goto LABEL_31;
            ++v41;
          }
          goto LABEL_46;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 240);
      }
      sub_15564C0(v46, 2 * v17);
      v32 = *(_DWORD *)(a1 + 264);
      if ( !v32 )
        goto LABEL_63;
      v16 = v53[0];
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 248);
      v35 = v33 & ((LODWORD(v53[0]) >> 9) ^ (LODWORD(v53[0]) >> 4));
      v20 = (_QWORD *)(v34 + 8LL * v35);
      v36 = *v20;
      v30 = *(_DWORD *)(a1 + 256) + 1;
      if ( *v20 == v53[0] )
        goto LABEL_31;
      v37 = 1;
      v38 = 0;
      while ( v36 != -8 )
      {
        if ( !v38 && v36 == -16 )
          v38 = v20;
        v35 = v33 & (v37 + v35);
        v20 = (_QWORD *)(v34 + 8LL * v35);
        v36 = *v20;
        if ( v53[0] == *v20 )
          goto LABEL_31;
        ++v37;
      }
LABEL_46:
      if ( v38 )
        v20 = v38;
LABEL_31:
      *(_DWORD *)(a1 + 256) = v30;
      if ( *v20 != -8 )
        --*(_DWORD *)(a1 + 260);
      *v20 = v16;
      v31 = *(_BYTE **)(a1 + 280);
      if ( v31 == *(_BYTE **)(a1 + 288) )
      {
        sub_15406B0(a1 + 272, v31, v53);
      }
      else
      {
        if ( v31 )
        {
          *(_QWORD *)v31 = v53[0];
          v31 = *(_BYTE **)(a1 + 280);
        }
        *(_QWORD *)(a1 + 280) = v31 + 8;
      }
LABEL_11:
      v24 = v53;
      v55 = 0;
      v25 = v53;
      v26 = &v49;
      v54 = sub_1548A60;
      v27 = sub_1548A30;
      if ( ((unsigned __int8)sub_1548A30 & 1) == 0 )
        goto LABEL_13;
      while ( 1 )
      {
        v27 = *(__int64 (__fastcall **)(__int64))((char *)v27 + *v26 - 1);
LABEL_13:
        if ( (unsigned __int8)v27((__int64)v26) )
          break;
        while ( 1 )
        {
          v28 = v25[3];
          v27 = (__int64 (__fastcall *)(__int64))v25[2];
          v24 += 2;
          v25 = v24;
          v26 = (__int64 *)((char *)&v49 + v28);
          if ( ((unsigned __int8)v27 & 1) != 0 )
            break;
          if ( (unsigned __int8)v27((__int64)v26) )
            goto LABEL_16;
        }
      }
LABEL_16:
      ;
    }
    while ( v48 != v51 );
LABEL_17:
    if ( v48 == v52 )
    {
      result = v47;
      if ( v47 == v49 && v47 == v50 )
        break;
    }
  }
  if ( !a5 )
  {
    if ( !qword_4F9DF70 )
      sub_16C1EA0(&qword_4F9DF70, sub_1548AF0, sub_1548AD0);
    result = *(_QWORD *)qword_4F9DF70;
    *(_QWORD *)(a1 + 232) = *(_QWORD *)qword_4F9DF70;
  }
  return result;
}
