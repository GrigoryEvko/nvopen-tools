// Function: sub_E6FF20
// Address: 0xe6ff20
//
__int64 __fastcall sub_E6FF20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int128 a8,
        char a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        unsigned int a13)
{
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdx
  _QWORD v20[2]; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v21[3]; // [rsp+40h] [rbp-30h] BYREF
  unsigned int *v22; // [rsp+58h] [rbp-18h] BYREF

  v14 = *(_QWORD *)(a2 + 1744);
  v20[0] = a5;
  v15 = a2 + 1736;
  v21[0] = a3;
  v21[1] = a4;
  v20[1] = a6;
  if ( !v14 )
  {
    v16 = a2 + 1736;
LABEL_8:
    v22 = &a13;
    LODWORD(v16) = sub_E6FC80((_QWORD *)(a2 + 1728), v16, &v22);
    goto LABEL_9;
  }
  v16 = a2 + 1736;
  do
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(v14 + 16);
      v18 = *(_QWORD *)(v14 + 24);
      if ( *(_DWORD *)(v14 + 32) >= a13 )
        break;
      v14 = *(_QWORD *)(v14 + 24);
      if ( !v18 )
        goto LABEL_6;
    }
    v16 = v14;
    v14 = *(_QWORD *)(v14 + 16);
  }
  while ( v17 );
LABEL_6:
  if ( v15 == v16 || a13 < *(_DWORD *)(v16 + 32) )
    goto LABEL_8;
LABEL_9:
  sub_E798E0(
    a1,
    v16 + 40,
    (unsigned int)v21,
    (unsigned int)v20,
    *(unsigned __int16 *)(a2 + 1904),
    a7,
    *(_OWORD *)&_mm_loadu_si128((const __m128i *)&a8),
    a9,
    a10,
    a11,
    a12);
  return a1;
}
