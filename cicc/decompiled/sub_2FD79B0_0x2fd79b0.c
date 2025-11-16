// Function: sub_2FD79B0
// Address: 0x2fd79b0
//
__int64 __fastcall sub_2FD79B0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 result; // rax

  v1 = *a1;
  if ( !*a1 )
    BUG();
  if ( (*(_BYTE *)v1 & 4) != 0 )
  {
    result = *(_QWORD *)(v1 + 8);
    *a1 = result;
  }
  else
  {
    while ( (*(_BYTE *)(v1 + 44) & 8) != 0 )
      v1 = *(_QWORD *)(v1 + 8);
    result = *(_QWORD *)(v1 + 8);
    *a1 = result;
  }
  return result;
}
