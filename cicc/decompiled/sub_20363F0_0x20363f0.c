// Function: sub_20363F0
// Address: 0x20363f0
//
__int64 __fastcall sub_20363F0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  int v4; // r12d
  char v5; // cl
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v8; // eax
  _DWORD *v9; // r13
  int v10; // edx
  char v11; // r9
  __int64 v12; // r10
  int v13; // r8d
  int v14; // esi
  unsigned int v15; // eax
  __int64 v16; // rcx
  int v17; // edx
  unsigned int v19; // esi
  unsigned int v20; // esi
  unsigned int v21; // eax
  __int64 v22; // rdi
  int v23; // edx
  unsigned int v24; // r8d
  int v25; // eax
  unsigned int v26; // eax
  int v27; // edx
  unsigned int v28; // edi
  int v29; // r9d
  _DWORD *v30; // r8
  int v31; // r11d
  __int64 v32; // rcx
  int v33; // eax
  unsigned int v34; // edx
  int v35; // esi
  __int64 v36; // rcx
  int v37; // edx
  unsigned int v38; // eax
  int v39; // esi
  int v40; // r8d
  _DWORD *v41; // rdi
  int v42; // eax
  int v43; // edx
  __int64 v44; // r14
  int v45; // r8d
  __int64 v46[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_200F8F0(a1, a2, a3);
  v5 = *(_BYTE *)(a1 + 1216) & 1;
  if ( v5 )
  {
    v6 = a1 + 1224;
    v7 = 7;
  }
  else
  {
    v19 = *(_DWORD *)(a1 + 1232);
    v6 = *(_QWORD *)(a1 + 1224);
    if ( !v19 )
    {
      v26 = *(_DWORD *)(a1 + 1216);
      ++*(_QWORD *)(a1 + 1208);
      v9 = 0;
      v27 = (v26 >> 1) + 1;
LABEL_20:
      v28 = 3 * v19;
      goto LABEL_21;
    }
    v7 = v19 - 1;
  }
  v8 = v7 & (37 * v4);
  v9 = (_DWORD *)(v6 + 8LL * v8);
  v10 = *v9;
  if ( v4 == *v9 )
    goto LABEL_4;
  v29 = 1;
  v30 = 0;
  while ( v10 != -1 )
  {
    if ( !v30 && v10 == -2 )
      v30 = v9;
    v8 = v7 & (v29 + v8);
    v9 = (_DWORD *)(v6 + 8LL * v8);
    v10 = *v9;
    if ( v4 == *v9 )
      goto LABEL_4;
    ++v29;
  }
  v26 = *(_DWORD *)(a1 + 1216);
  v28 = 24;
  v19 = 8;
  if ( v30 )
    v9 = v30;
  ++*(_QWORD *)(a1 + 1208);
  v27 = (v26 >> 1) + 1;
  if ( !v5 )
  {
    v19 = *(_DWORD *)(a1 + 1232);
    goto LABEL_20;
  }
LABEL_21:
  if ( 4 * v27 >= v28 )
  {
    sub_20108A0(a1 + 1208, 2 * v19);
    if ( (*(_BYTE *)(a1 + 1216) & 1) != 0 )
    {
      v32 = a1 + 1224;
      v33 = 7;
    }
    else
    {
      v42 = *(_DWORD *)(a1 + 1232);
      v32 = *(_QWORD *)(a1 + 1224);
      if ( !v42 )
        goto LABEL_78;
      v33 = v42 - 1;
    }
    v34 = v33 & (37 * v4);
    v9 = (_DWORD *)(v32 + 8LL * v34);
    v35 = *v9;
    if ( v4 != *v9 )
    {
      v45 = 1;
      v41 = 0;
      while ( v35 != -1 )
      {
        if ( !v41 && v35 == -2 )
          v41 = v9;
        v34 = v33 & (v45 + v34);
        v9 = (_DWORD *)(v32 + 8LL * v34);
        v35 = *v9;
        if ( v4 == *v9 )
          goto LABEL_43;
        ++v45;
      }
      goto LABEL_49;
    }
LABEL_43:
    v26 = *(_DWORD *)(a1 + 1216);
    goto LABEL_23;
  }
  if ( v19 - *(_DWORD *)(a1 + 1220) - v27 <= v19 >> 3 )
  {
    sub_20108A0(a1 + 1208, v19);
    if ( (*(_BYTE *)(a1 + 1216) & 1) != 0 )
    {
      v36 = a1 + 1224;
      v37 = 7;
      goto LABEL_46;
    }
    v43 = *(_DWORD *)(a1 + 1232);
    v36 = *(_QWORD *)(a1 + 1224);
    if ( v43 )
    {
      v37 = v43 - 1;
LABEL_46:
      v38 = v37 & (37 * v4);
      v9 = (_DWORD *)(v36 + 8LL * v38);
      v39 = *v9;
      if ( v4 != *v9 )
      {
        v40 = 1;
        v41 = 0;
        while ( v39 != -1 )
        {
          if ( !v41 && v39 == -2 )
            v41 = v9;
          v38 = v37 & (v40 + v38);
          v9 = (_DWORD *)(v36 + 8LL * v38);
          v39 = *v9;
          if ( v4 == *v9 )
            goto LABEL_43;
          ++v40;
        }
LABEL_49:
        if ( v41 )
          v9 = v41;
        goto LABEL_43;
      }
      goto LABEL_43;
    }
LABEL_78:
    *(_DWORD *)(a1 + 1216) = (2 * (*(_DWORD *)(a1 + 1216) >> 1) + 2) | *(_DWORD *)(a1 + 1216) & 1;
    BUG();
  }
LABEL_23:
  *(_DWORD *)(a1 + 1216) = (2 * (v26 >> 1) + 2) | v26 & 1;
  if ( *v9 != -1 )
    --*(_DWORD *)(a1 + 1220);
  *v9 = v4;
  v9[1] = 0;
LABEL_4:
  sub_200D1B0(a1, v9 + 1);
  v11 = *(_BYTE *)(a1 + 352) & 1;
  if ( v11 )
  {
    v12 = a1 + 360;
    v13 = 7;
  }
  else
  {
    v20 = *(_DWORD *)(a1 + 368);
    v12 = *(_QWORD *)(a1 + 360);
    if ( !v20 )
    {
      v21 = *(_DWORD *)(a1 + 352);
      ++*(_QWORD *)(a1 + 344);
      v22 = 0;
      v23 = (v21 >> 1) + 1;
LABEL_13:
      v24 = 3 * v20;
      goto LABEL_14;
    }
    v13 = v20 - 1;
  }
  v14 = v9[1];
  v15 = v13 & (37 * v14);
  v16 = v12 + 24LL * v15;
  v17 = *(_DWORD *)v16;
  if ( *(_DWORD *)v16 == v14 )
    return *(_QWORD *)(v16 + 8);
  v31 = 1;
  v22 = 0;
  while ( v17 != -1 )
  {
    if ( v22 || v17 != -2 )
      v16 = v22;
    v15 = v13 & (v31 + v15);
    v44 = v12 + 24LL * v15;
    v17 = *(_DWORD *)v44;
    if ( v14 == *(_DWORD *)v44 )
      return *(_QWORD *)(v44 + 8);
    ++v31;
    v22 = v16;
    v16 = v12 + 24LL * v15;
  }
  v21 = *(_DWORD *)(a1 + 352);
  v24 = 24;
  v20 = 8;
  if ( !v22 )
    v22 = v16;
  ++*(_QWORD *)(a1 + 344);
  v23 = (v21 >> 1) + 1;
  if ( !v11 )
  {
    v20 = *(_DWORD *)(a1 + 368);
    goto LABEL_13;
  }
LABEL_14:
  if ( v24 <= 4 * v23 )
  {
    v20 *= 2;
    goto LABEL_27;
  }
  if ( v20 - *(_DWORD *)(a1 + 356) - v23 <= v20 >> 3 )
  {
LABEL_27:
    sub_200F500(a1 + 344, v20);
    sub_2032230(a1 + 344, v9 + 1, v46);
    v22 = v46[0];
    v21 = *(_DWORD *)(a1 + 352);
  }
  *(_DWORD *)(a1 + 352) = (2 * (v21 >> 1) + 2) | v21 & 1;
  if ( *(_DWORD *)v22 != -1 )
    --*(_DWORD *)(a1 + 356);
  v25 = v9[1];
  *(_QWORD *)(v22 + 8) = 0;
  *(_DWORD *)(v22 + 16) = 0;
  *(_DWORD *)v22 = v25;
  return 0;
}
