// Function: sub_284F450
// Address: 0x284f450
//
_BYTE **__fastcall sub_284F450(_BYTE **a1, _BYTE **a2, __int64 a3, __int64 a4)
{
  _BYTE **i; // r12
  _BYTE *v7; // r13
  __int64 *v8; // rax

  for ( i = a1; a2 != i; i += 4 )
  {
    v7 = *i;
    if ( **i > 0x1Cu && sub_D97040(a4, *((_QWORD *)v7 + 1)) )
    {
      v8 = sub_DD8400(a4, (__int64)v7);
      if ( *((_WORD *)v8 + 12) == 8 && a3 == v8[6] )
        break;
    }
  }
  return i;
}
