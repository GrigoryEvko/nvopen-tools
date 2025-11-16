// Function: sub_13D0DF0
// Address: 0x13d0df0
//
__int64 __fastcall sub_13D0DF0(_QWORD *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax

  if ( *((_BYTE *)a1 + 8) != 16 )
    return sub_1643320(*a1);
  v1 = a1[4];
  v2 = sub_1643320(*a1);
  return sub_16463B0(v2, (unsigned int)v1);
}
