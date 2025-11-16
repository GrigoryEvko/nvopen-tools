// Function: sub_8EE740
// Address: 0x8ee740
//
unsigned __int64 __fastcall sub_8EE740(char *a1, signed int a2, int a3)
{
  int v4; // r8d
  int v5; // ecx
  signed int v6; // r9d

  if ( a3 <= 0 )
    return sub_8EE610(a1, a2, a3, 0);
  if ( a2 < a3 )
    return sub_8EE610(a1, a2, a3, 0);
  v4 = (unsigned __int8)a1[(a3 - 1) >> 3];
  if ( !_bittest(&v4, ((_BYTE)a3 - 1) & 7) )
    return sub_8EE610(a1, a2, a3, 0);
  v5 = sub_8EE5A0((__int64)a1, a3 - 1);
  if ( v5 )
  {
    v5 = 1;
  }
  else if ( v6 > a3 )
  {
    v5 = ((1 << (a3 & 7)) & (unsigned __int8)a1[a3 >> 3]) != 0;
  }
  return sub_8EE610(a1, v6, a3, v5);
}
