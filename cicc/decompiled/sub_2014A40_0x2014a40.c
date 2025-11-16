// Function: sub_2014A40
// Address: 0x2014a40
//
__int64 __fastcall sub_2014A40(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __m128i *a5, const __m128i *a6)
{
  __int64 *v8; // rax
  bool v9; // zf
  int v10; // ebx
  char v11; // cl
  __int64 v12; // rdx
  int v13; // esi
  unsigned int v14; // eax
  _DWORD *v15; // r13
  int v16; // edi
  __int64 result; // rax
  unsigned int v18; // esi
  unsigned int v19; // eax
  int v20; // edx
  unsigned int v21; // edi
  int v22; // r9d
  _DWORD *v23; // r8
  __int64 v24; // rcx
  int v25; // eax
  unsigned int v26; // edx
  int v27; // esi
  __int64 v28; // rcx
  int v29; // edx
  unsigned int v30; // eax
  int v31; // esi
  int v32; // r8d
  _DWORD *v33; // rdi
  int v34; // eax
  int v35; // edx
  int v36; // r8d
  unsigned __int64 v37; // [rsp+0h] [rbp-30h] BYREF
  __m128i *v38; // [rsp+8h] [rbp-28h]

  v37 = a4;
  v38 = a5;
  v8 = sub_2010420(a1, a4, a3, a4, a5, a6);
  v9 = *((_DWORD *)v8 + 7) == -3;
  v37 = (unsigned __int64)v8;
  if ( v9 )
    sub_2010110(a1, (__int64)&v37);
  v10 = sub_200F8F0(a1, a2, a3);
  v11 = *(_BYTE *)(a1 + 832) & 1;
  if ( v11 )
  {
    v12 = a1 + 840;
    v13 = 7;
  }
  else
  {
    v18 = *(_DWORD *)(a1 + 848);
    v12 = *(_QWORD *)(a1 + 840);
    if ( !v18 )
    {
      v19 = *(_DWORD *)(a1 + 832);
      v15 = 0;
      ++*(_QWORD *)(a1 + 824);
      v20 = (v19 >> 1) + 1;
LABEL_10:
      v21 = 3 * v18;
      goto LABEL_11;
    }
    v13 = v18 - 1;
  }
  v14 = v13 & (37 * v10);
  v15 = (_DWORD *)(v12 + 8LL * v14);
  v16 = *v15;
  if ( v10 == *v15 )
    goto LABEL_6;
  v22 = 1;
  v23 = 0;
  while ( v16 != -1 )
  {
    if ( !v23 && v16 == -2 )
      v23 = v15;
    v14 = v13 & (v22 + v14);
    v15 = (_DWORD *)(v12 + 8LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_6;
    ++v22;
  }
  v19 = *(_DWORD *)(a1 + 832);
  v21 = 24;
  v18 = 8;
  if ( v23 )
    v15 = v23;
  ++*(_QWORD *)(a1 + 824);
  v20 = (v19 >> 1) + 1;
  if ( !v11 )
  {
    v18 = *(_DWORD *)(a1 + 848);
    goto LABEL_10;
  }
LABEL_11:
  if ( 4 * v20 >= v21 )
  {
    sub_20108A0(a1 + 824, 2 * v18);
    if ( (*(_BYTE *)(a1 + 832) & 1) != 0 )
    {
      v24 = a1 + 840;
      v25 = 7;
    }
    else
    {
      v34 = *(_DWORD *)(a1 + 848);
      v24 = *(_QWORD *)(a1 + 840);
      if ( !v34 )
        goto LABEL_54;
      v25 = v34 - 1;
    }
    v26 = v25 & (37 * v10);
    v15 = (_DWORD *)(v24 + 8LL * v26);
    v27 = *v15;
    if ( v10 != *v15 )
    {
      v36 = 1;
      v33 = 0;
      while ( v27 != -1 )
      {
        if ( !v33 && v27 == -2 )
          v33 = v15;
        v26 = v25 & (v36 + v26);
        v15 = (_DWORD *)(v24 + 8LL * v26);
        v27 = *v15;
        if ( v10 == *v15 )
          goto LABEL_25;
        ++v36;
      }
      goto LABEL_31;
    }
LABEL_25:
    v19 = *(_DWORD *)(a1 + 832);
    goto LABEL_13;
  }
  if ( v18 - *(_DWORD *)(a1 + 836) - v20 <= v18 >> 3 )
  {
    sub_20108A0(a1 + 824, v18);
    if ( (*(_BYTE *)(a1 + 832) & 1) != 0 )
    {
      v28 = a1 + 840;
      v29 = 7;
      goto LABEL_28;
    }
    v35 = *(_DWORD *)(a1 + 848);
    v28 = *(_QWORD *)(a1 + 840);
    if ( v35 )
    {
      v29 = v35 - 1;
LABEL_28:
      v30 = v29 & (37 * v10);
      v15 = (_DWORD *)(v28 + 8LL * v30);
      v31 = *v15;
      if ( v10 != *v15 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -1 )
        {
          if ( v31 == -2 && !v33 )
            v33 = v15;
          v30 = v29 & (v32 + v30);
          v15 = (_DWORD *)(v28 + 8LL * v30);
          v31 = *v15;
          if ( v10 == *v15 )
            goto LABEL_25;
          ++v32;
        }
LABEL_31:
        if ( v33 )
          v15 = v33;
        goto LABEL_25;
      }
      goto LABEL_25;
    }
LABEL_54:
    *(_DWORD *)(a1 + 832) = (2 * (*(_DWORD *)(a1 + 832) >> 1) + 2) | *(_DWORD *)(a1 + 832) & 1;
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 832) = (2 * (v19 >> 1) + 2) | v19 & 1;
  if ( *v15 != -1 )
    --*(_DWORD *)(a1 + 836);
  *v15 = v10;
  v15[1] = 0;
LABEL_6:
  result = sub_200F8F0(a1, v37, (__int64)v38);
  v15[1] = result;
  return result;
}
