// Function: sub_2015400
// Address: 0x2015400
//
unsigned __int64 __fastcall sub_2015400(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i *a5,
        const __m128i *a6)
{
  __int64 *v9; // rax
  bool v10; // zf
  int v11; // ebx
  char v12; // cl
  __int64 v13; // rdx
  int v14; // esi
  unsigned int v15; // eax
  _DWORD *v16; // r14
  int v17; // edi
  unsigned __int64 result; // rax
  unsigned int v19; // esi
  unsigned int v20; // eax
  int v21; // edx
  unsigned int v22; // edi
  int v23; // r9d
  _DWORD *v24; // r8
  __int64 v25; // rcx
  int v26; // eax
  unsigned int v27; // edx
  int v28; // esi
  __int64 v29; // rcx
  int v30; // edx
  unsigned int v31; // eax
  int v32; // esi
  int v33; // r8d
  _DWORD *v34; // rdi
  int v35; // eax
  int v36; // edx
  int v37; // r8d
  unsigned __int64 v38; // [rsp+0h] [rbp-30h] BYREF
  __m128i *v39; // [rsp+8h] [rbp-28h]

  v38 = a4;
  v39 = a5;
  v9 = sub_2010420(a1, a4, a3, a4, a5, a6);
  v10 = *((_DWORD *)v9 + 7) == -3;
  v38 = (unsigned __int64)v9;
  if ( v10 )
    sub_2010110(a1, (__int64)&v38);
  v11 = sub_200F8F0(a1, a2, a3);
  v12 = *(_BYTE *)(a1 + 1216) & 1;
  if ( v12 )
  {
    v13 = a1 + 1224;
    v14 = 7;
  }
  else
  {
    v19 = *(_DWORD *)(a1 + 1232);
    v13 = *(_QWORD *)(a1 + 1224);
    if ( !v19 )
    {
      v20 = *(_DWORD *)(a1 + 1216);
      v16 = 0;
      ++*(_QWORD *)(a1 + 1208);
      v21 = (v20 >> 1) + 1;
LABEL_10:
      v22 = 3 * v19;
      goto LABEL_11;
    }
    v14 = v19 - 1;
  }
  v15 = v14 & (37 * v11);
  v16 = (_DWORD *)(v13 + 8LL * v15);
  v17 = *v16;
  if ( v11 == *v16 )
    goto LABEL_6;
  v23 = 1;
  v24 = 0;
  while ( v17 != -1 )
  {
    if ( !v24 && v17 == -2 )
      v24 = v16;
    v15 = v14 & (v23 + v15);
    v16 = (_DWORD *)(v13 + 8LL * v15);
    v17 = *v16;
    if ( v11 == *v16 )
      goto LABEL_6;
    ++v23;
  }
  v20 = *(_DWORD *)(a1 + 1216);
  v22 = 24;
  v19 = 8;
  if ( v24 )
    v16 = v24;
  ++*(_QWORD *)(a1 + 1208);
  v21 = (v20 >> 1) + 1;
  if ( !v12 )
  {
    v19 = *(_DWORD *)(a1 + 1232);
    goto LABEL_10;
  }
LABEL_11:
  if ( 4 * v21 >= v22 )
  {
    sub_20108A0(a1 + 1208, 2 * v19);
    if ( (*(_BYTE *)(a1 + 1216) & 1) != 0 )
    {
      v25 = a1 + 1224;
      v26 = 7;
    }
    else
    {
      v35 = *(_DWORD *)(a1 + 1232);
      v25 = *(_QWORD *)(a1 + 1224);
      if ( !v35 )
        goto LABEL_54;
      v26 = v35 - 1;
    }
    v27 = v26 & (37 * v11);
    v16 = (_DWORD *)(v25 + 8LL * v27);
    v28 = *v16;
    if ( v11 != *v16 )
    {
      v37 = 1;
      v34 = 0;
      while ( v28 != -1 )
      {
        if ( !v34 && v28 == -2 )
          v34 = v16;
        v27 = v26 & (v37 + v27);
        v16 = (_DWORD *)(v25 + 8LL * v27);
        v28 = *v16;
        if ( v11 == *v16 )
          goto LABEL_25;
        ++v37;
      }
      goto LABEL_31;
    }
LABEL_25:
    v20 = *(_DWORD *)(a1 + 1216);
    goto LABEL_13;
  }
  if ( v19 - *(_DWORD *)(a1 + 1220) - v21 <= v19 >> 3 )
  {
    sub_20108A0(a1 + 1208, v19);
    if ( (*(_BYTE *)(a1 + 1216) & 1) != 0 )
    {
      v29 = a1 + 1224;
      v30 = 7;
      goto LABEL_28;
    }
    v36 = *(_DWORD *)(a1 + 1232);
    v29 = *(_QWORD *)(a1 + 1224);
    if ( v36 )
    {
      v30 = v36 - 1;
LABEL_28:
      v31 = v30 & (37 * v11);
      v16 = (_DWORD *)(v29 + 8LL * v31);
      v32 = *v16;
      if ( v11 != *v16 )
      {
        v33 = 1;
        v34 = 0;
        while ( v32 != -1 )
        {
          if ( v32 == -2 && !v34 )
            v34 = v16;
          v31 = v30 & (v33 + v31);
          v16 = (_DWORD *)(v29 + 8LL * v31);
          v32 = *v16;
          if ( v11 == *v16 )
            goto LABEL_25;
          ++v33;
        }
LABEL_31:
        if ( v34 )
          v16 = v34;
        goto LABEL_25;
      }
      goto LABEL_25;
    }
LABEL_54:
    *(_DWORD *)(a1 + 1216) = (2 * (*(_DWORD *)(a1 + 1216) >> 1) + 2) | *(_DWORD *)(a1 + 1216) & 1;
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 1216) = (2 * (v20 >> 1) + 2) | v20 & 1;
  if ( *v16 != -1 )
    --*(_DWORD *)(a1 + 1220);
  *v16 = v11;
  v16[1] = 0;
LABEL_6:
  v16[1] = sub_200F8F0(a1, v38, (__int64)v39);
  result = v38;
  *(_DWORD *)(v38 + 64) = *(_DWORD *)(a2 + 64);
  return result;
}
