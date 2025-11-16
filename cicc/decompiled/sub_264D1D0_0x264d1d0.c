// Function: sub_264D1D0
// Address: 0x264d1d0
//
void __fastcall sub_264D1D0(__int64 a1, int a2)
{
  unsigned int v2; // eax

  if ( a2 )
  {
    v2 = sub_AF1560(4 * a2 / 3u + 1);
    ++*(_QWORD *)a1;
    if ( *(_DWORD *)(a1 + 24) < v2 )
      sub_A08C50(a1, v2);
  }
  else
  {
    ++*(_QWORD *)a1;
  }
}
