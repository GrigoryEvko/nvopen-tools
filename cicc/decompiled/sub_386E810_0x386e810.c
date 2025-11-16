// Function: sub_386E810
// Address: 0x386e810
//
__int64 __fastcall sub_386E810(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // rax
  __int64 v13; // r9
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r11
  _QWORD *v17; // rax
  bool v18; // zf
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 result; // rax
  int v22; // edx
  int v23; // ebx

  v12 = *(unsigned int *)(*(_QWORD *)a1 + 112LL);
  if ( !(_DWORD)v12 )
    return sub_386C880(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
  v13 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  v14 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (__int64 *)(v13 + 16LL * v14);
  v16 = *v15;
  if ( a2 != *v15 )
  {
    v22 = 1;
    while ( v16 != -8 )
    {
      v23 = v22 + 1;
      v14 = (v12 - 1) & (v22 + v14);
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( a2 == *v15 )
        goto LABEL_3;
      v22 = v23;
    }
    return sub_386C880(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
  }
LABEL_3:
  if ( v15 == (__int64 *)(v13 + 16 * v12) )
    return sub_386C880(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
  v17 = (_QWORD *)v15[1];
  if ( !v17 )
    return sub_386C880(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
  v19 = *v17 & 0xFFFFFFFFFFFFFFF8LL;
  v18 = v19 == 0;
  v20 = v19 - 48;
  result = 0;
  if ( !v18 )
    return v20;
  return result;
}
