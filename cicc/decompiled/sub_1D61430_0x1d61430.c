// Function: sub_1D61430
// Address: 0x1d61430
//
__int64 (__fastcall *__fastcall sub_1D61430(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4))(__int64 a1, __int64 a2, __m128 a3, double a4, double a5, double a6, double a7, double a8, double a9, __m128 a10, __int64 a11, int *a12, __int64 a13, __int64 a14, _QWORD *a15)
{
  __int64 *v6; // rax
  unsigned __int8 *v7; // rbx
  unsigned __int8 v8; // al
  __int64 v9; // r15
  char v10; // r14
  bool v11; // dl
  __int64 v12; // rdi
  bool v13; // zf
  __int64 **v14; // rax
  __int64 *v15; // rdi
  __int64 v16; // rax
  void *v17; // r8
  __int64 v19; // r12
  __int64 (*v20)(); // rax
  __int64 v21; // rdi
  unsigned int v22; // r12d
  unsigned __int8 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // r8
  unsigned int v27; // esi
  __int64 **v28; // rcx
  __int64 *v29; // r10
  unsigned __int64 v30; // rdx
  _QWORD *v31; // rax
  __int64 v32; // rdi
  _QWORD *v33; // rax
  __int64 v34; // rax
  unsigned int v35; // ecx
  unsigned __int64 v36; // rax
  int v37; // eax
  int v38; // ecx
  char v39; // [rsp+4h] [rbp-3Ch]
  unsigned int v40; // [rsp+4h] [rbp-3Ch]
  __int64 v41; // [rsp+8h] [rbp-38h]
  __int64 v42; // [rsp+8h] [rbp-38h]
  unsigned int v43; // [rsp+8h] [rbp-38h]
  int v44; // [rsp+8h] [rbp-38h]

  if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
    v6 = (__int64 *)*(a1 - 1);
  else
    v6 = &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
  v7 = (unsigned __int8 *)*v6;
  v8 = *(_BYTE *)(*v6 + 16);
  if ( v8 <= 0x17u || *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 )
    return 0;
  v9 = *a1;
  v10 = *((_BYTE *)a1 + 16);
  if ( v8 == 61 )
    return sub_1D5F790;
  v11 = v10 == 62;
  if ( v8 == 62 && v10 == 62 )
    return sub_1D5F790;
  if ( (unsigned int)v8 - 35 > 0x11 || v8 > 0x2Fu )
    goto LABEL_14;
  v12 = 0x80A800000000LL;
  if ( !_bittest64(&v12, v8) )
    goto LABEL_18;
  v41 = a4;
  if ( v10 != 62 )
  {
    v13 = !sub_15F2370((__int64)v7);
    v8 = v7[16];
    if ( v13 )
    {
      v11 = v10 == 62;
      a4 = v41;
      goto LABEL_14;
    }
LABEL_24:
    if ( v8 != 60 )
      goto LABEL_27;
LABEL_25:
    if ( !sub_13A0E30(a2, (__int64)v7) )
    {
      v8 = v7[16];
LABEL_27:
      v17 = sub_1D5F790;
      if ( (unsigned __int8)(v8 - 60) <= 2u )
        return (__int64 (__fastcall *)(__int64, __int64, __m128, double, double, double, double, double, double, __m128, __int64, int *, __int64, __int64, _QWORD *))v17;
      goto LABEL_28;
    }
    return 0;
  }
  v13 = !sub_15F2380((__int64)v7);
  v8 = v7[16];
  if ( !v13 )
    goto LABEL_24;
  a4 = v41;
  v11 = v10 == 62;
LABEL_14:
  if ( (unsigned __int8)(v8 - 50) <= 1u )
    goto LABEL_27;
  if ( v8 == 52 )
  {
    v21 = *(_QWORD *)(sub_13CF970((__int64)v7) + 24);
    if ( *(_BYTE *)(v21 + 16) != 13 )
      return 0;
    v22 = *(_DWORD *)(v21 + 32);
    if ( v22 <= 0x40
       ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) == *(_QWORD *)(v21 + 24)
       : v22 == (unsigned int)sub_16A58F0(v21 + 24) )
    {
      return 0;
    }
    goto LABEL_28;
  }
  if ( v8 == 48 && !v11 )
  {
LABEL_28:
    v19 = *((_QWORD *)v7 + 1);
    goto LABEL_29;
  }
LABEL_18:
  if ( v8 != 47 )
  {
    v39 = v11;
    v42 = a4;
    if ( v8 != 60 )
      return 0;
    v14 = (__int64 **)sub_13CF970((__int64)v7);
    v15 = *v14;
    v16 = **v14;
    if ( *(_BYTE *)(v16 + 8) != 11 )
      return 0;
    if ( *(_DWORD *)(v16 + 8) >> 8 > *(_DWORD *)(v9 + 8) >> 8 )
      return 0;
    v24 = *((_BYTE *)v15 + 16);
    if ( v24 <= 0x17u )
      return 0;
    v25 = *(unsigned int *)(v42 + 24);
    if ( (_DWORD)v25 )
    {
      v26 = *(_QWORD *)(v42 + 8);
      v27 = (v25 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v28 = (__int64 **)(v26 + 16LL * v27);
      v29 = *v28;
      if ( v15 == *v28 )
      {
LABEL_46:
        if ( v28 != (__int64 **)(v26 + 16 * v25) && v39 == (((__int64)v28[1] >> 1) & 3) )
        {
          v30 = (unsigned __int64)v28[1] & 0xFFFFFFFFFFFFFFF8LL;
          if ( v30 )
          {
LABEL_49:
            if ( *(_DWORD *)(*(_QWORD *)v7 + 8LL) >> 8 < *(_DWORD *)(v30 + 8) >> 8 )
              return 0;
            goto LABEL_25;
          }
        }
      }
      else
      {
        v38 = 1;
        while ( v29 != (__int64 *)-8LL )
        {
          v27 = (v25 - 1) & (v38 + v27);
          v44 = v38 + 1;
          v28 = (__int64 **)(v26 + 16LL * v27);
          v29 = *v28;
          if ( v15 == *v28 )
            goto LABEL_46;
          v38 = v44;
        }
      }
    }
    if ( v10 == 62 )
    {
      if ( v24 != 62 )
        return 0;
    }
    else if ( v24 != 61 )
    {
      return 0;
    }
    v30 = **(_QWORD **)sub_13CF970((__int64)v15);
    goto LABEL_49;
  }
  v19 = *((_QWORD *)v7 + 1);
  if ( !v19 || *(_QWORD *)(v19 + 8) )
    return 0;
  v31 = sub_1648700(*((_QWORD *)v7 + 1));
  if ( *((_BYTE *)v31 + 16) <= 0x17u )
    BUG();
  v32 = v31[1];
  if ( !v32 )
    return 0;
  if ( *(_QWORD *)(v32 + 8) )
    return 0;
  v33 = sub_1648700(v32);
  if ( *((_BYTE *)v33 + 16) != 50 )
    return 0;
  v34 = *(_QWORD *)(sub_13CF970((__int64)v33) + 24);
  if ( *(_BYTE *)(v34 + 16) != 13 )
    return 0;
  v35 = *(_DWORD *)(v34 + 32);
  v43 = *(_DWORD *)(*(_QWORD *)v7 + 8LL) >> 8;
  if ( v35 > 0x40 )
  {
    v40 = *(_DWORD *)(v34 + 32);
    v37 = sub_16A57B0(v34 + 24);
    v35 = v40;
  }
  else
  {
    v36 = *(_QWORD *)(v34 + 24);
    if ( v36 )
    {
      _BitScanReverse64(&v36, v36);
      LODWORD(v36) = v36 ^ 0x3F;
    }
    else
    {
      LODWORD(v36) = 64;
    }
    v37 = v35 + v36 - 64;
  }
  if ( v43 < v35 - v37 )
    return 0;
LABEL_29:
  if ( !v19 || *(_QWORD *)(v19 + 8) )
  {
    v20 = *(__int64 (**)())(*(_QWORD *)a3 + 784LL);
    if ( v20 == sub_1D5A3F0 || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v20)(a3, v9, *(_QWORD *)v7) )
      return 0;
  }
  v17 = sub_1D608F0;
  if ( v10 != 62 )
    return (__int64 (__fastcall *)(__int64, __int64, __m128, double, double, double, double, double, double, __m128, __int64, int *, __int64, __int64, _QWORD *))sub_1D608E0;
  return (__int64 (__fastcall *)(__int64, __int64, __m128, double, double, double, double, double, double, __m128, __int64, int *, __int64, __int64, _QWORD *))v17;
}
