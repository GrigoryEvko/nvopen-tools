// Function: sub_38520A0
// Address: 0x38520a0
//
bool __fastcall sub_38520A0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 v6; // rbx
  int v7; // edx
  int v8; // edi
  __int64 v9; // rsi
  unsigned int v10; // eax
  __int64 v11; // rcx
  _QWORD *v12; // rax
  __int64 v13; // r12
  __int64 v14; // r13
  int v15; // eax
  unsigned int v16; // esi
  unsigned int v17; // r8d
  __int64 v18; // rdi
  unsigned int v19; // r9d
  __int64 *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  _QWORD *v24; // rdx
  int v25; // eax
  int v26; // ecx
  int v27; // eax
  int v28; // ecx
  int v29; // r9d
  __int64 v30; // rdi
  _QWORD *v31; // r8
  unsigned int v32; // eax
  __int64 v33; // rsi
  int v34; // eax
  int v35; // eax
  __int64 v36; // rdi
  __int64 v37; // r9
  __int64 v38; // rsi
  int v39; // r10d
  _QWORD *v40; // r8
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  int v43; // eax
  int v44; // eax
  int v45; // eax
  int v46; // r9d
  __int64 v47; // r8
  unsigned int v48; // r10d
  __int64 v49; // rdi
  int v50; // esi
  _QWORD *v51; // rcx
  int v52; // eax
  int v53; // eax
  __int64 v54; // r8
  __int64 v55; // r10
  __int64 v56; // rdi
  int v57; // esi
  _QWORD *v58; // rcx
  __int64 *v59; // [rsp+8h] [rbp-68h]
  int i; // [rsp+1Ch] [rbp-54h]
  unsigned int v61; // [rsp+20h] [rbp-50h]
  __int64 *v62; // [rsp+20h] [rbp-50h]
  __int64 v63; // [rsp+28h] [rbp-48h]
  int v64; // [rsp+28h] [rbp-48h]
  unsigned int v65; // [rsp+28h] [rbp-48h]
  unsigned int v66; // [rsp+28h] [rbp-48h]
  __int64 v68; // [rsp+38h] [rbp-38h]
  _QWORD *v69; // [rsp+38h] [rbp-38h]
  unsigned int v70; // [rsp+38h] [rbp-38h]

  v6 = a1;
  if ( a2 == a1 )
    return a2 == v6;
  do
  {
    v12 = sub_1648700(v6);
    v13 = *a3;
    v14 = v12[5];
    v15 = *(_DWORD *)(*a3 + 288);
    if ( v15 )
    {
      v7 = v15 - 1;
      v8 = 1;
      v9 = *(_QWORD *)(v13 + 272);
      v10 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v11 = *(_QWORD *)(v9 + 8LL * v10);
      if ( v14 == v11 )
        goto LABEL_4;
      while ( v11 != -8 )
      {
        v10 = v7 & (v8 + v10);
        v11 = *(_QWORD *)(v9 + 8LL * v10);
        if ( v14 == v11 )
          goto LABEL_4;
        ++v8;
      }
    }
    v16 = *(_DWORD *)(v13 + 344);
    if ( !v16 )
    {
      ++*(_QWORD *)(v13 + 320);
      goto LABEL_28;
    }
    v17 = v16 - 1;
    v18 = *(_QWORD *)(v13 + 328);
    v19 = (v16 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v20 = (__int64 *)(v18 + 16LL * v19);
    v21 = *v20;
    if ( v14 == *v20 )
    {
      if ( v20[1] )
      {
        v22 = v20[1];
        v68 = *a4;
        goto LABEL_12;
      }
      return a2 == v6;
    }
    v63 = *v20;
    v24 = 0;
    v69 = (_QWORD *)(v18 + 16LL * (v17 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4))));
    v61 = (v16 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    for ( i = 1; ; ++i )
    {
      if ( v63 == -8 )
      {
        v25 = *(_DWORD *)(v13 + 336);
        if ( !v24 )
          v24 = v69;
        ++*(_QWORD *)(v13 + 320);
        v26 = v25 + 1;
        if ( 4 * (v25 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(v13 + 340) - v26 > v16 >> 3 )
          {
LABEL_24:
            *(_DWORD *)(v13 + 336) = v26;
            if ( *v24 != -8 )
              --*(_DWORD *)(v13 + 340);
            *v24 = v14;
            v24[1] = 0;
            return a2 == v6;
          }
          v70 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
          sub_1447B20(v13 + 320, v16);
          v34 = *(_DWORD *)(v13 + 344);
          if ( v34 )
          {
            v35 = v34 - 1;
            v36 = *(_QWORD *)(v13 + 328);
            LODWORD(v37) = v35 & v70;
            v26 = *(_DWORD *)(v13 + 336) + 1;
            v24 = (_QWORD *)(v36 + 16LL * (v35 & v70));
            v38 = *v24;
            if ( v14 != *v24 )
            {
              v39 = 1;
              v40 = 0;
              while ( v38 != -8 )
              {
                if ( v38 == -16 && !v40 )
                  v40 = v24;
                v37 = v35 & (unsigned int)(v37 + v39);
                v24 = (_QWORD *)(v36 + 16 * v37);
                v38 = *v24;
                if ( v14 == *v24 )
                  goto LABEL_24;
                ++v39;
              }
              if ( v40 )
                v24 = v40;
            }
            goto LABEL_24;
          }
LABEL_96:
          ++*(_DWORD *)(v13 + 336);
          BUG();
        }
LABEL_28:
        sub_1447B20(v13 + 320, 2 * v16);
        v27 = *(_DWORD *)(v13 + 344);
        if ( v27 )
        {
          v28 = v27 - 1;
          v29 = 1;
          v30 = *(_QWORD *)(v13 + 328);
          v31 = 0;
          v32 = (v27 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v24 = (_QWORD *)(v30 + 16LL * v32);
          v33 = *v24;
          if ( v14 == *v24 )
          {
LABEL_30:
            v26 = *(_DWORD *)(v13 + 336) + 1;
          }
          else
          {
            while ( v33 != -8 )
            {
              if ( v33 == -16 && !v31 )
                v31 = v24;
              v32 = v28 & (v29 + v32);
              v24 = (_QWORD *)(v30 + 16LL * v32);
              v33 = *v24;
              if ( v14 == *v24 )
                goto LABEL_30;
              ++v29;
            }
            v26 = *(_DWORD *)(v13 + 336) + 1;
            if ( v31 )
              v24 = v31;
          }
          goto LABEL_24;
        }
        goto LABEL_96;
      }
      if ( v24 || v63 != -16 )
        v69 = v24;
      v61 = v17 & (v61 + i);
      v59 = (__int64 *)(v18 + 16LL * v61);
      v63 = *v59;
      if ( v14 == *v59 )
        break;
      v24 = v69;
      v69 = (_QWORD *)(v18 + 16LL * v61);
    }
    v41 = (_QWORD *)(v18 + 16LL * (v17 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4))));
    if ( !v59[1] )
      return a2 == v6;
    v64 = 1;
    v68 = *a4;
    v42 = 0;
    while ( v21 != -8 )
    {
      if ( v21 != -16 || v42 )
        v41 = v42;
      v19 = v17 & (v64 + v19);
      v62 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v62;
      if ( v14 == *v62 )
      {
        v22 = v62[1];
        goto LABEL_12;
      }
      ++v64;
      v42 = v41;
      v41 = (_QWORD *)(v18 + 16LL * v19);
    }
    if ( !v42 )
      v42 = v41;
    v43 = *(_DWORD *)(v13 + 336);
    ++*(_QWORD *)(v13 + 320);
    v44 = v43 + 1;
    if ( 4 * v44 >= 3 * v16 )
    {
      v65 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
      sub_1447B20(v13 + 320, 2 * v16);
      v45 = *(_DWORD *)(v13 + 344);
      if ( v45 )
      {
        v46 = v45 - 1;
        v47 = *(_QWORD *)(v13 + 328);
        v48 = v46 & v65;
        v44 = *(_DWORD *)(v13 + 336) + 1;
        v42 = (_QWORD *)(v47 + 16LL * (v46 & v65));
        v49 = *v42;
        if ( v14 != *v42 )
        {
          v50 = 1;
          v51 = 0;
          while ( v49 != -8 )
          {
            if ( !v51 && v49 == -16 )
              v51 = v42;
            v48 = v46 & (v50 + v48);
            v42 = (_QWORD *)(v47 + 16LL * v48);
            v49 = *v42;
            if ( v14 == *v42 )
              goto LABEL_48;
            ++v50;
          }
          if ( v51 )
            v42 = v51;
        }
        goto LABEL_48;
      }
      goto LABEL_97;
    }
    if ( v16 - *(_DWORD *)(v13 + 340) - v44 <= v16 >> 3 )
    {
      v66 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
      sub_1447B20(v13 + 320, v16);
      v52 = *(_DWORD *)(v13 + 344);
      if ( v52 )
      {
        v53 = v52 - 1;
        v54 = *(_QWORD *)(v13 + 328);
        LODWORD(v55) = v53 & v66;
        v42 = (_QWORD *)(v54 + 16LL * (v53 & v66));
        v56 = *v42;
        if ( v14 == *v42 )
        {
LABEL_60:
          v44 = *(_DWORD *)(v13 + 336) + 1;
        }
        else
        {
          v57 = 1;
          v58 = 0;
          while ( v56 != -8 )
          {
            if ( v56 == -16 && !v58 )
              v58 = v42;
            v55 = v53 & (unsigned int)(v55 + v57);
            v42 = (_QWORD *)(v54 + 16 * v55);
            v56 = *v42;
            if ( v14 == *v42 )
              goto LABEL_60;
            ++v57;
          }
          v44 = *(_DWORD *)(v13 + 336) + 1;
          if ( v58 )
            v42 = v58;
        }
        goto LABEL_48;
      }
LABEL_97:
      ++*(_DWORD *)(v13 + 336);
      BUG();
    }
LABEL_48:
    *(_DWORD *)(v13 + 336) = v44;
    if ( *v42 != -8 )
      --*(_DWORD *)(v13 + 340);
    *v42 = v14;
    v22 = 0;
    v42[1] = 0;
LABEL_12:
    if ( v68 == v22 )
      break;
    do
LABEL_4:
      v6 = *(_QWORD *)(v6 + 8);
    while ( v6 && (unsigned __int8)(*((_BYTE *)sub_1648700(v6) + 16) - 25) > 9u );
  }
  while ( a2 != v6 );
  return a2 == v6;
}
