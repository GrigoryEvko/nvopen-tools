// Function: sub_272E800
// Address: 0x272e800
//
void __fastcall sub_272E800(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r9

  if ( a2 - a1 <= 2352 )
  {
    sub_272DD00(a1, a2, a3, a4, a5, a6);
  }
  else
  {
    v6 = 168 * ((__int64)(0xCF3CF3CF3CF3CF3DLL * ((a2 - a1) >> 3)) >> 1);
    sub_272E800(a1, a1 + v6);
    sub_272E800(a1 + v6, a2);
    sub_272E520(a1, a1 + v6, a2, 0xCF3CF3CF3CF3CF3DLL * (v6 >> 3), 0xCF3CF3CF3CF3CF3DLL * ((a2 - (a1 + v6)) >> 3), v7);
  }
}
