// Function: sub_13143B0
// Address: 0x13143b0
//
__int64 __fastcall sub_13143B0(_QWORD *a1, _QWORD *a2)
{
  unsigned __int64 v2; // rax
  _QWORD *v3; // rdx
  __int64 v4; // rcx
  unsigned __int64 v6; // rcx
  __int64 v7; // rax

  v2 = a1[8];
  v3 = a1 + 8;
  LODWORD(v4) = 0;
  if ( !v2 )
  {
    do
    {
      v4 = (unsigned int)(v4 + 1);
      v2 = v3[v4];
    }
    while ( !v2 );
    LODWORD(v4) = (_DWORD)v4 << 6;
  }
  if ( !_BitScanForward64(&v2, v2) )
    LODWORD(v2) = -1;
  v6 = (unsigned int)(v4 + v2);
  v3[v6 >> 6] ^= 1LL << v6;
  v7 = *a2 * v6;
  *a1 -= 0x10000000LL;
  return a1[1] + v7;
}
