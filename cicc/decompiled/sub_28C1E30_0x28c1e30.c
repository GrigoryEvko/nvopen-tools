// Function: sub_28C1E30
// Address: 0x28c1e30
//
__int64 __fastcall sub_28C1E30(__int64 a1, unsigned __int8 *a2, _BYTE *a3, _QWORD *a4, _QWORD *a5)
{
  int v5; // eax
  __int64 v7; // rax
  __int64 v8; // rax

  v5 = *a2;
  if ( v5 == 42 )
  {
    if ( *a3 != 42 )
      return 0;
  }
  else
  {
    if ( v5 != 46 )
      BUG();
    if ( *a3 != 46 )
      return 0;
  }
  v7 = *((_QWORD *)a3 - 8);
  if ( !v7 )
    return 0;
  *a4 = v7;
  v8 = *((_QWORD *)a3 - 4);
  if ( !v8 )
    return 0;
  *a5 = v8;
  return 1;
}
