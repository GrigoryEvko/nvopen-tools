// Function: sub_19C7030
// Address: 0x19c7030
//
__int64 __fastcall sub_19C7030(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v14; // rax
  _DWORD *v15; // rax
  double v16; // xmm4_8
  double v17; // xmm5_8

  v14 = sub_13FD000(*(_QWORD *)(a1 + 296));
  if ( v14 && sub_1AFD990(v14, "llvm.loop.unroll.full", 21) )
  {
    v15 = *(_DWORD **)(a1 + 272);
    if ( a4
      && *(_BYTE *)(a4 + 16) == 27
      && dword_4FB3720 >= ((*(_DWORD *)(a4 + 20) & 0xFFFFFFFu) >> 1) - 1
      && (unsigned int)(3 * *(_DWORD *)(a1 + 280)) >> 1 > v15[2] )
    {
      goto LABEL_7;
    }
  }
  else
  {
    v15 = *(_DWORD **)(a1 + 272);
  }
  if ( !*v15 )
    return 0;
LABEL_7:
  if ( (unsigned int)sub_1C105D0(*(_QWORD *)(a1 + 184), a2, 0) )
    return 0;
  sub_19C4800(a1, a2, a3, *(__int64 **)(a1 + 296), a4, a5, a6, a7, a8, v16, v17, a11, a12);
  return 1;
}
