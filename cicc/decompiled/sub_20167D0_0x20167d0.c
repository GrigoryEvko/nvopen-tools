// Function: sub_20167D0
// Address: 0x20167d0
//
__int64 __fastcall sub_20167D0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i *a5,
        const __m128i *a6,
        unsigned __int64 a7,
        __int64 a8)
{
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __m128i *v14; // r8
  const __m128i *v15; // r9
  bool v16; // zf
  __int64 *v17; // rax
  int v18; // ebx
  char v19; // al
  __int64 v20; // rdi
  int v21; // esi
  unsigned int v22; // edx
  __int64 v23; // r13
  int v24; // ecx
  int v25; // eax
  unsigned __int64 v26; // rdx
  __int64 result; // rax
  unsigned int v28; // esi
  unsigned int v29; // edi
  int v30; // edx
  unsigned int v31; // ecx
  int v32; // r9d
  __int64 v33; // r8
  __int64 v34; // rsi
  int v35; // eax
  unsigned int v36; // edx
  int v37; // ecx
  __int64 v38; // rsi
  int v39; // edx
  unsigned int v40; // ecx
  int v41; // eax
  int v42; // r8d
  __int64 v43; // rdi
  int v44; // eax
  int v45; // edx
  int v46; // r8d
  unsigned __int64 v47; // [rsp+0h] [rbp-30h] BYREF
  __m128i *v48; // [rsp+8h] [rbp-28h]

  v47 = a4;
  v48 = a5;
  v11 = sub_2010420(a1, a4, a3, a4, a5, a6);
  v16 = *((_DWORD *)v11 + 7) == -3;
  v47 = (unsigned __int64)v11;
  if ( v16 )
    sub_2010110(a1, (__int64)&v47);
  v17 = sub_2010420(a1, a7, v12, v13, v14, v15);
  v16 = *((_DWORD *)v17 + 7) == -3;
  a7 = (unsigned __int64)v17;
  if ( v16 )
    sub_2010110(a1, (__int64)&a7);
  v18 = sub_200F8F0(a1, a2, a3);
  v19 = *(_BYTE *)(a1 + 1104) & 1;
  if ( v19 )
  {
    v20 = a1 + 1112;
    v21 = 7;
  }
  else
  {
    v28 = *(_DWORD *)(a1 + 1120);
    v20 = *(_QWORD *)(a1 + 1112);
    if ( !v28 )
    {
      v29 = *(_DWORD *)(a1 + 1104);
      v23 = 0;
      ++*(_QWORD *)(a1 + 1096);
      v30 = (v29 >> 1) + 1;
LABEL_12:
      v31 = 3 * v28;
      goto LABEL_13;
    }
    v21 = v28 - 1;
  }
  v22 = v21 & (37 * v18);
  v23 = v20 + 12LL * v22;
  v24 = *(_DWORD *)v23;
  if ( v18 == *(_DWORD *)v23 )
    goto LABEL_8;
  v32 = 1;
  v33 = 0;
  while ( v24 != -1 )
  {
    if ( !v33 && v24 == -2 )
      v33 = v23;
    v22 = v21 & (v32 + v22);
    v23 = v20 + 12LL * v22;
    v24 = *(_DWORD *)v23;
    if ( v18 == *(_DWORD *)v23 )
      goto LABEL_8;
    ++v32;
  }
  v29 = *(_DWORD *)(a1 + 1104);
  v31 = 24;
  v28 = 8;
  if ( v33 )
    v23 = v33;
  ++*(_QWORD *)(a1 + 1096);
  v30 = (v29 >> 1) + 1;
  if ( !v19 )
  {
    v28 = *(_DWORD *)(a1 + 1120);
    goto LABEL_12;
  }
LABEL_13:
  if ( 4 * v30 >= v31 )
  {
    sub_2015860(a1 + 1096, 2 * v28);
    if ( (*(_BYTE *)(a1 + 1104) & 1) != 0 )
    {
      v34 = a1 + 1112;
      v35 = 7;
    }
    else
    {
      v44 = *(_DWORD *)(a1 + 1120);
      v34 = *(_QWORD *)(a1 + 1112);
      if ( !v44 )
        goto LABEL_56;
      v35 = v44 - 1;
    }
    v36 = v35 & (37 * v18);
    v23 = v34 + 12LL * v36;
    v37 = *(_DWORD *)v23;
    if ( v18 != *(_DWORD *)v23 )
    {
      v46 = 1;
      v43 = 0;
      while ( v37 != -1 )
      {
        if ( !v43 && v37 == -2 )
          v43 = v23;
        v36 = v35 & (v46 + v36);
        v23 = v34 + 12LL * v36;
        v37 = *(_DWORD *)v23;
        if ( v18 == *(_DWORD *)v23 )
          goto LABEL_27;
        ++v46;
      }
      goto LABEL_33;
    }
LABEL_27:
    v29 = *(_DWORD *)(a1 + 1104);
    goto LABEL_15;
  }
  if ( v28 - *(_DWORD *)(a1 + 1108) - v30 <= v28 >> 3 )
  {
    sub_2015860(a1 + 1096, v28);
    if ( (*(_BYTE *)(a1 + 1104) & 1) != 0 )
    {
      v38 = a1 + 1112;
      v39 = 7;
      goto LABEL_30;
    }
    v45 = *(_DWORD *)(a1 + 1120);
    v38 = *(_QWORD *)(a1 + 1112);
    if ( v45 )
    {
      v39 = v45 - 1;
LABEL_30:
      v40 = v39 & (37 * v18);
      v23 = v38 + 12LL * v40;
      v41 = *(_DWORD *)v23;
      if ( v18 != *(_DWORD *)v23 )
      {
        v42 = 1;
        v43 = 0;
        while ( v41 != -1 )
        {
          if ( v41 == -2 && !v43 )
            v43 = v23;
          v40 = v39 & (v42 + v40);
          v23 = v38 + 12LL * v40;
          v41 = *(_DWORD *)v23;
          if ( v18 == *(_DWORD *)v23 )
            goto LABEL_27;
          ++v42;
        }
LABEL_33:
        if ( v43 )
          v23 = v43;
        goto LABEL_27;
      }
      goto LABEL_27;
    }
LABEL_56:
    *(_DWORD *)(a1 + 1104) = (2 * (*(_DWORD *)(a1 + 1104) >> 1) + 2) | *(_DWORD *)(a1 + 1104) & 1;
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 1104) = (2 * (v29 >> 1) + 2) | v29 & 1;
  if ( *(_DWORD *)v23 != -1 )
    --*(_DWORD *)(a1 + 1108);
  *(_DWORD *)v23 = v18;
  *(_QWORD *)(v23 + 4) = 0;
LABEL_8:
  *(_DWORD *)(v23 + 4) = sub_200F8F0(a1, v47, (__int64)v48);
  v25 = sub_200F8F0(a1, a7, a8);
  v26 = v47;
  *(_DWORD *)(v23 + 8) = v25;
  result = *(unsigned int *)(a2 + 64);
  *(_DWORD *)(v26 + 64) = result;
  *(_DWORD *)(a7 + 64) = result;
  return result;
}
