// Function: sub_21EDC30
// Address: 0x21edc30
//
__int64 __fastcall sub_21EDC30(__int64 a1, int a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  unsigned int v8; // esi
  int v9; // eax
  __int64 v10; // r8
  unsigned int v11; // edx
  __int64 *v12; // rdi
  int v13; // ecx
  unsigned int v14; // esi
  int v15; // edx
  __int64 v16; // r8
  unsigned int v17; // eax
  __int64 *v18; // rdi
  int v19; // ecx
  unsigned int v20; // esi
  int v21; // ecx
  __int64 v22; // r8
  unsigned int v23; // edx
  __int64 *v24; // rax
  int v25; // edi
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 v28; // rsi
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // r9
  __int64 v32; // r14
  __int64 v33; // rbx
  __int64 result; // rax
  __int64 v35; // r13
  unsigned int v36; // esi
  unsigned int v37; // r9d
  __int64 v38; // r8
  unsigned int v39; // edx
  __int64 *v40; // rcx
  int v41; // edi
  unsigned int v42; // eax
  unsigned __int64 v43; // rdx
  __int64 v44; // rax
  unsigned int v45; // esi
  __int64 v46; // r8
  unsigned int v47; // edx
  __int64 *v48; // rdi
  int v49; // ecx
  int v50; // r11d
  __int64 *v51; // r10
  int v52; // edi
  int v53; // ecx
  int v54; // eax
  int v55; // r10d
  int i; // r11d
  int v57; // r11d
  __int64 *v58; // r10
  int v59; // ecx
  int v60; // ecx
  int v61; // r11d
  __int64 *v62; // r10
  int v63; // eax
  int v64; // ecx
  int v65; // r11d
  __int64 *v66; // r10
  int v67; // edi
  int v68; // edi
  int v69; // r11d
  __int64 *v70; // r10
  int v71; // edi
  int v72; // ecx
  int v73; // r10d
  int v74; // r10d
  int *v75; // r11
  unsigned int v76; // eax
  unsigned int v77; // [rsp+10h] [rbp-70h]
  int v78; // [rsp+18h] [rbp-68h] BYREF
  int v79[3]; // [rsp+1Ch] [rbp-64h] BYREF
  unsigned int v80; // [rsp+28h] [rbp-58h] BYREF
  int v81; // [rsp+2Ch] [rbp-54h] BYREF
  __int64 *v82[10]; // [rsp+30h] [rbp-50h] BYREF

  v78 = a3;
  v7 = *(_QWORD *)(a4 + 24);
  v79[0] = a2;
  if ( v7 == *(_QWORD *)(a5 + 24) )
  {
    sub_21EAD50(v82, (__int64 *)(a1 + 112), v7);
    if ( !(unsigned __int8)sub_21EBD20(a1, v79[0], v82[2][1]) )
      goto LABEL_4;
  }
  v8 = *(_DWORD *)(a1 + 288);
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 264);
LABEL_82:
    v8 *= 2;
    goto LABEL_83;
  }
  v9 = v78;
  v10 = *(_QWORD *)(a1 + 272);
  v11 = (v8 - 1) & (37 * v78);
  v12 = (__int64 *)(v10 + 4LL * v11);
  v13 = *(_DWORD *)v12;
  if ( v78 == *(_DWORD *)v12 )
    goto LABEL_4;
  v69 = 1;
  v70 = 0;
  while ( v13 != -1 )
  {
    if ( v13 != -2 || v70 )
      v12 = v70;
    v11 = (v8 - 1) & (v69 + v11);
    v13 = *(_DWORD *)(v10 + 4LL * v11);
    if ( v78 == v13 )
      goto LABEL_4;
    ++v69;
    v70 = v12;
    v12 = (__int64 *)(v10 + 4LL * v11);
  }
  if ( !v70 )
    v70 = v12;
  v71 = *(_DWORD *)(a1 + 280);
  ++*(_QWORD *)(a1 + 264);
  v72 = v71 + 1;
  if ( 4 * (v71 + 1) >= 3 * v8 )
    goto LABEL_82;
  if ( v8 - *(_DWORD *)(a1 + 284) - v72 <= v8 >> 3 )
  {
LABEL_83:
    sub_136B240(a1 + 264, v8);
    sub_1DF91F0(a1 + 264, &v78, v82);
    v70 = v82[0];
    v9 = v78;
    v72 = *(_DWORD *)(a1 + 280) + 1;
  }
  *(_DWORD *)(a1 + 280) = v72;
  if ( *(_DWORD *)v70 != -1 )
    --*(_DWORD *)(a1 + 284);
  *(_DWORD *)v70 = v9;
LABEL_4:
  v14 = *(_DWORD *)(a1 + 224);
  if ( !v14 )
  {
    ++*(_QWORD *)(a1 + 200);
LABEL_70:
    v14 *= 2;
    goto LABEL_71;
  }
  v15 = v78;
  v16 = *(_QWORD *)(a1 + 208);
  v17 = (v14 - 1) & (37 * v78);
  v18 = (__int64 *)(v16 + 4LL * v17);
  v19 = *(_DWORD *)v18;
  if ( v78 == *(_DWORD *)v18 )
    goto LABEL_6;
  v61 = 1;
  v62 = 0;
  while ( v19 != -1 )
  {
    if ( v19 != -2 || v62 )
      v18 = v62;
    v17 = (v14 - 1) & (v61 + v17);
    v19 = *(_DWORD *)(v16 + 4LL * v17);
    if ( v78 == v19 )
      goto LABEL_6;
    ++v61;
    v62 = v18;
    v18 = (__int64 *)(v16 + 4LL * v17);
  }
  v63 = *(_DWORD *)(a1 + 216);
  if ( !v62 )
    v62 = v18;
  ++*(_QWORD *)(a1 + 200);
  v64 = v63 + 1;
  if ( 4 * (v63 + 1) >= 3 * v14 )
    goto LABEL_70;
  if ( v14 - *(_DWORD *)(a1 + 220) - v64 <= v14 >> 3 )
  {
LABEL_71:
    sub_136B240(a1 + 200, v14);
    sub_1DF91F0(a1 + 200, &v78, v82);
    v62 = v82[0];
    v15 = v78;
    v64 = *(_DWORD *)(a1 + 216) + 1;
  }
  *(_DWORD *)(a1 + 216) = v64;
  if ( *(_DWORD *)v62 != -1 )
    --*(_DWORD *)(a1 + 220);
  *(_DWORD *)v62 = v15;
LABEL_6:
  v20 = *(_DWORD *)(a1 + 320);
  if ( !v20 )
  {
    ++*(_QWORD *)(a1 + 296);
LABEL_67:
    v20 *= 2;
    goto LABEL_68;
  }
  v21 = v79[0];
  v22 = *(_QWORD *)(a1 + 304);
  v23 = (v20 - 1) & (37 * v79[0]);
  v24 = (__int64 *)(v22 + 16LL * v23);
  v25 = *(_DWORD *)v24;
  if ( v79[0] == *(_DWORD *)v24 )
    goto LABEL_8;
  v65 = 1;
  v66 = 0;
  while ( v25 != -1 )
  {
    if ( v25 == -2 && !v66 )
      v66 = v24;
    v23 = (v20 - 1) & (v65 + v23);
    v24 = (__int64 *)(v22 + 16LL * v23);
    v25 = *(_DWORD *)v24;
    if ( v79[0] == *(_DWORD *)v24 )
      goto LABEL_8;
    ++v65;
  }
  v67 = *(_DWORD *)(a1 + 312);
  if ( v66 )
    v24 = v66;
  ++*(_QWORD *)(a1 + 296);
  v68 = v67 + 1;
  if ( 4 * v68 >= 3 * v20 )
    goto LABEL_67;
  if ( v20 - *(_DWORD *)(a1 + 316) - v68 <= v20 >> 3 )
  {
LABEL_68:
    sub_1DA3420(a1 + 296, v20);
    sub_21EAE00(a1 + 296, v79, v82);
    v24 = v82[0];
    v21 = v79[0];
    v68 = *(_DWORD *)(a1 + 312) + 1;
  }
  *(_DWORD *)(a1 + 312) = v68;
  if ( *(_DWORD *)v24 != -1 )
    --*(_DWORD *)(a1 + 316);
  *(_DWORD *)v24 = v21;
  v24[1] = 0;
LABEL_8:
  v26 = *(_QWORD *)(a4 + 24);
  v24[1] = v26;
  v27 = *(_DWORD *)(a1 + 136);
  v28 = *(_QWORD *)(a1 + 120);
  if ( v27 )
  {
    v29 = (v27 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v30 = (__int64 *)(v28 + 16LL * v29);
    v31 = *v30;
    if ( v26 == *v30 )
      goto LABEL_10;
    v54 = 1;
    while ( v31 != -8 )
    {
      v73 = v54 + 1;
      v29 = (v27 - 1) & (v54 + v29);
      v30 = (__int64 *)(v28 + 16LL * v29);
      v31 = *v30;
      if ( v26 == *v30 )
        goto LABEL_10;
      v54 = v73;
    }
  }
  v30 = (__int64 *)(v28 + 16LL * v27);
LABEL_10:
  v32 = v30[1];
  v33 = *(_QWORD *)(a4 + 32);
  result = 5LL * *(unsigned int *)(a4 + 40);
  v35 = v33 + 40LL * *(unsigned int *)(a4 + 40);
  if ( v35 != v33 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)v33 )
        goto LABEL_12;
      result = *(unsigned int *)(v33 + 8);
      v80 = result;
      if ( (int)result >= 0 )
        goto LABEL_12;
      v36 = *(_DWORD *)(a1 + 80);
      v81 = result;
      if ( !v36 )
        goto LABEL_12;
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 64);
      v39 = (v36 - 1) & (37 * result);
      v40 = (__int64 *)(v38 + 8LL * v39);
      v41 = *(_DWORD *)v40;
      if ( (_DWORD)result != *(_DWORD *)v40 )
      {
        v77 = (v36 - 1) & (37 * result);
        v55 = *(_DWORD *)v40;
        for ( i = 1; ; ++i )
        {
          if ( v55 == -1 )
            goto LABEL_12;
          v77 = v37 & (v77 + i);
          v55 = *(_DWORD *)(v38 + 8LL * v77);
          if ( (_DWORD)result == v55 )
            break;
        }
        v57 = 1;
        v58 = 0;
        while ( v41 != -1 )
        {
          if ( v58 || v41 != -2 )
            v40 = v58;
          v74 = v57 + 1;
          v39 = v37 & (v57 + v39);
          v75 = (int *)(v38 + 8LL * v39);
          v41 = *v75;
          if ( (_DWORD)result == *v75 )
          {
            v76 = v75[1];
            v43 = v76 & 0x3F;
            v44 = 8LL * (v76 >> 6);
            goto LABEL_18;
          }
          v57 = v74;
          v58 = v40;
          v40 = (__int64 *)(v38 + 8LL * v39);
        }
        if ( !v58 )
          v58 = v40;
        v59 = *(_DWORD *)(a1 + 72);
        ++*(_QWORD *)(a1 + 56);
        v60 = v59 + 1;
        if ( 4 * v60 >= 3 * v36 )
        {
          v36 *= 2;
        }
        else if ( v36 - *(_DWORD *)(a1 + 76) - v60 > v36 >> 3 )
        {
LABEL_45:
          *(_DWORD *)(a1 + 72) = v60;
          if ( *(_DWORD *)v58 != -1 )
            --*(_DWORD *)(a1 + 76);
          *(_DWORD *)v58 = result;
          v43 = 0;
          v44 = 0;
          *((_DWORD *)v58 + 1) = 0;
          goto LABEL_18;
        }
        sub_1BFDD60(a1 + 56, v36);
        sub_1BFD720(a1 + 56, &v81, v82);
        v58 = v82[0];
        LODWORD(result) = v81;
        v60 = *(_DWORD *)(a1 + 72) + 1;
        goto LABEL_45;
      }
      v42 = *((_DWORD *)v40 + 1);
      v43 = v42 & 0x3F;
      v44 = 8LL * (v42 >> 6);
LABEL_18:
      result = *(_QWORD *)(*(_QWORD *)(v32 + 24) + v44);
      if ( !_bittest64(&result, v43) )
        goto LABEL_12;
      v45 = *(_DWORD *)(a1 + 256);
      if ( !v45 )
      {
        ++*(_QWORD *)(a1 + 232);
LABEL_85:
        v45 *= 2;
LABEL_86:
        sub_136B240(a1 + 232, v45);
        sub_1DF91F0(a1 + 232, (int *)&v80, v82);
        v51 = v82[0];
        result = v80;
        v53 = *(_DWORD *)(a1 + 248) + 1;
        goto LABEL_27;
      }
      result = v80;
      v46 = *(_QWORD *)(a1 + 240);
      v47 = (v45 - 1) & (37 * v80);
      v48 = (__int64 *)(v46 + 4LL * v47);
      v49 = *(_DWORD *)v48;
      if ( *(_DWORD *)v48 == v80 )
      {
LABEL_12:
        v33 += 40;
        if ( v35 == v33 )
          return result;
      }
      else
      {
        v50 = 1;
        v51 = 0;
        while ( v49 != -1 )
        {
          if ( v49 != -2 || v51 )
            v48 = v51;
          v47 = (v45 - 1) & (v50 + v47);
          v49 = *(_DWORD *)(v46 + 4LL * v47);
          if ( v80 == v49 )
            goto LABEL_12;
          ++v50;
          v51 = v48;
          v48 = (__int64 *)(v46 + 4LL * v47);
        }
        if ( !v51 )
          v51 = v48;
        v52 = *(_DWORD *)(a1 + 248);
        ++*(_QWORD *)(a1 + 232);
        v53 = v52 + 1;
        if ( 4 * (v52 + 1) >= 3 * v45 )
          goto LABEL_85;
        if ( v45 - *(_DWORD *)(a1 + 252) - v53 <= v45 >> 3 )
          goto LABEL_86;
LABEL_27:
        *(_DWORD *)(a1 + 248) = v53;
        if ( *(_DWORD *)v51 != -1 )
          --*(_DWORD *)(a1 + 252);
        v33 += 40;
        *(_DWORD *)v51 = result;
        if ( v35 == v33 )
          return result;
      }
    }
  }
  return result;
}
