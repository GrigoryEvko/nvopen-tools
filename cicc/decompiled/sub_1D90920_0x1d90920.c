// Function: sub_1D90920
// Address: 0x1d90920
//
__int64 __fastcall sub_1D90920(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v15; // rax
  _BYTE *v16; // rax
  double v17; // xmm4_8
  double v18; // xmm5_8
  unsigned int v19; // edx
  unsigned __int8 v20; // si
  unsigned __int8 v21; // cl

  if ( (*(_BYTE *)(a2 + 19) & 0x40) == 0 )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_13:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4FC3606 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_13;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4FC3606);
  v16 = *(_BYTE **)(sub_1D8F610(v15, a2) + 8);
  v19 = (unsigned __int8)v16[49];
  v20 = v16[48];
  v21 = v16[51];
  if ( !(_BYTE)v19 || !v20 )
    return sub_1D902E0(a2, v20, v19, v21, a3, a4, a5, a6, v17, v18, a9, a10);
  if ( v21 )
  {
    v20 = v16[51];
    return sub_1D902E0(a2, v20, v19, v21, a3, a4, a5, a6, v17, v18, a9, a10);
  }
  return 0;
}
