// Function: sub_B59AF0
// Address: 0xb59af0
//
__int64 __fastcall sub_B59AF0(__int64 a1)
{
  unsigned __int8 *v1; // rax
  unsigned int v2; // r8d

  v1 = (unsigned __int8 *)sub_B595C0(a1);
  v2 = 1;
  if ( v1 )
    LOBYTE(v2) = (unsigned int)*v1 - 12 <= 1;
  return v2;
}
