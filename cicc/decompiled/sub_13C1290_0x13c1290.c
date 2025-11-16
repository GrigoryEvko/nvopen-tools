// Function: sub_13C1290
// Address: 0x13c1290
//
__int64 __fastcall sub_13C1290(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rax

  v2 = sub_13C1210(a1, a2);
  v3 = 63;
  if ( v2 && (v3 = 4, v4 = *v2 & 3, (_BYTE)v4) )
    return (v4 & 2) == 0 ? 61 : 63;
  else
    return v3;
}
