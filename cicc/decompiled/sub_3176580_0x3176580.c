// Function: sub_3176580
// Address: 0x3176580
//
__int64 __fastcall sub_3176580(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  int v4; // r14d
  _QWORD *v5; // rax
  __int64 v6; // rcx
  _QWORD *v7; // rdx
  char v8; // cl
  __int64 v9; // r12
  __int128 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rsi
  __int64 *v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 *v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 *v30; // rax
  _QWORD *v32; // r13
  _QWORD *v33; // r14
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rax
  unsigned int v37; // eax
  _QWORD *v38; // r14
  _QWORD *v39; // r13
  __int64 v40; // rsi
  __int128 v42; // [rsp+10h] [rbp-110h] BYREF
  __int128 v43; // [rsp+20h] [rbp-100h]
  __int64 v44; // [rsp+30h] [rbp-F0h]
  __int128 v45; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v46; // [rsp+50h] [rbp-D0h]
  __int64 v47; // [rsp+58h] [rbp-C8h]
  __int64 v48; // [rsp+60h] [rbp-C0h]
  const char *v49; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v50; // [rsp+78h] [rbp-A8h] BYREF
  __int128 v51; // [rsp+80h] [rbp-A0h]
  __int64 i; // [rsp+90h] [rbp-90h]
  __int64 v53; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD *v54; // [rsp+A8h] [rbp-78h]
  __int64 v55; // [rsp+B0h] [rbp-70h]
  unsigned int v56; // [rsp+B8h] [rbp-68h]
  _QWORD *v57; // [rsp+C8h] [rbp-58h]
  unsigned int v58; // [rsp+D8h] [rbp-48h]
  char v59; // [rsp+E0h] [rbp-40h]

  v3 = *(_DWORD *)(a1 + 172);
  v53 = 0;
  v4 = v3 + 1 - *(_DWORD *)(a1 + 176);
  v56 = 128;
  v5 = (_QWORD *)sub_C7D670(0x2000, 8);
  v55 = 0;
  v54 = v5;
  v50 = 2;
  v7 = v5 + 1024;
  v49 = (const char *)&unk_49DD7B0;
  *(_QWORD *)&v51 = 0;
  *((_QWORD *)&v51 + 1) = -4096;
  for ( i = 0; v7 != v5; v5 += 8 )
  {
    if ( v5 )
    {
      v8 = v50;
      v5[2] = 0;
      v5[3] = -4096;
      *v5 = &unk_49DD7B0;
      v5[1] = v8 & 6;
      v6 = i;
      v5[4] = i;
    }
  }
  v59 = 0;
  LODWORD(v45) = v4;
  LOWORD(v48) = 265;
  v9 = sub_F4BFF0(a2, (__int64)&v53, 0, v6);
  *(_QWORD *)&v10 = sub_BD5D20(a2);
  v42 = v10;
  *(_QWORD *)&v43 = ".specialized.";
  LOWORD(v44) = 773;
  v51 = v45;
  v49 = (const char *)&v42;
  LOWORD(i) = 2306;
  sub_BD6B50((unsigned __int8 *)v9, &v49);
  sub_3174F80(v9);
  if ( v59 )
  {
    v37 = v58;
    v59 = 0;
    if ( v58 )
    {
      v38 = v57;
      v39 = &v57[2 * v58];
      do
      {
        if ( *v38 != -4096 && *v38 != -8192 )
        {
          v40 = v38[1];
          if ( v40 )
            sub_B91220((__int64)(v38 + 1), v40);
        }
        v38 += 2;
      }
      while ( v39 != v38 );
      v37 = v58;
    }
    sub_C7D6A0((__int64)v57, 16LL * v37, 8);
  }
  v11 = v56;
  if ( v56 )
  {
    v32 = v54;
    *(_QWORD *)&v45 = &unk_49DD7B0;
    v49 = (const char *)&unk_49DD7B0;
    v33 = &v54[8 * (unsigned __int64)v56];
    v34 = -4096;
    *((_QWORD *)&v45 + 1) = 2;
    v46 = 0;
    v47 = -4096;
    v48 = 0;
    v50 = 2;
    *(_QWORD *)&v51 = 0;
    *((_QWORD *)&v51 + 1) = -8192;
    i = 0;
    while ( 1 )
    {
      v35 = v32[3];
      if ( v35 != v34 )
      {
        v34 = *((_QWORD *)&v51 + 1);
        if ( v35 != *((_QWORD *)&v51 + 1) )
        {
          v36 = v32[7];
          if ( v36 != 0 && v36 != -4096 && v36 != -8192 )
          {
            sub_BD60C0(v32 + 5);
            v35 = v32[3];
          }
          v34 = v35;
        }
      }
      *v32 = &unk_49DB368;
      if ( v34 != -4096 && v34 != 0 && v34 != -8192 )
        sub_BD60C0(v32 + 1);
      v32 += 8;
      if ( v33 == v32 )
        break;
      v34 = v47;
    }
    v49 = (const char *)&unk_49DB368;
    if ( *((_QWORD *)&v51 + 1) != -4096 && *((_QWORD *)&v51 + 1) != 0 && *((_QWORD *)&v51 + 1) != -8192 )
      sub_BD60C0(&v50);
    *(_QWORD *)&v45 = &unk_49DB368;
    if ( v47 != -4096 && v47 != 0 && v47 != -8192 )
      sub_BD60C0((_QWORD *)&v45 + 1);
    v11 = v56;
  }
  sub_C7D6A0((__int64)v54, v11 << 6, 8);
  *(_WORD *)(v9 + 32) = *(_WORD *)(v9 + 32) & 0xBCC0 | 0x4007;
  sub_2A6E290(*(__int64 **)a1, v9, (unsigned int *)(a3 + 8), v12);
  v17 = *(_QWORD *)(v9 + 80);
  if ( v17 )
    v17 -= 24;
  sub_2A63F40(*(__int64 **)a1, v17, v13, v14, v15, v16);
  sub_2A64130(*(__int64 **)a1, v9, v18, v19, v20, v21);
  sub_2A73720(*(__int64 **)a1, v9, v22, v23, v24, v25);
  if ( !*(_BYTE *)(a1 + 180) )
  {
LABEL_16:
    sub_C8CC70(a1 + 152, v9, (__int64)v26, v27, v28, v29);
    return v9;
  }
  v30 = *(__int64 **)(a1 + 160);
  v27 = *(unsigned int *)(a1 + 172);
  v26 = &v30[v27];
  if ( v30 == v26 )
  {
LABEL_15:
    if ( (unsigned int)v27 < *(_DWORD *)(a1 + 168) )
    {
      *(_DWORD *)(a1 + 172) = v27 + 1;
      *v26 = v9;
      ++*(_QWORD *)(a1 + 152);
      return v9;
    }
    goto LABEL_16;
  }
  while ( v9 != *v30 )
  {
    if ( v26 == ++v30 )
      goto LABEL_15;
  }
  return v9;
}
