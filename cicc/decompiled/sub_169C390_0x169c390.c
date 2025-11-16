// Function: sub_169C390
// Address: 0x169c390
//
__int64 __fastcall sub_169C390(__int64 a1, __int16 **a2, int a3, unsigned int a4)
{
  int v6; // eax
  int v7; // edx

  v6 = **a2 - ((*a2)[1] - *((_DWORD *)*a2 + 1));
  v7 = ~v6;
  if ( ~v6 < a3 )
    v7 = a3;
  if ( v7 > v6 )
    LOWORD(v7) = **a2 - ((*a2)[1] - (*a2)[2]);
  *((_WORD *)a2 + 8) += v7;
  sub_1698EC0(a2, a4, 0);
  if ( (*((_BYTE *)a2 + 18) & 7) == 1 )
    sub_169C2C0((__int64)a2);
  sub_1698450(a1, (__int64)a2);
  return a1;
}
