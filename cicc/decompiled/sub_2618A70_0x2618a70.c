// Function: sub_2618A70
// Address: 0x2618a70
//
__int64 __fastcall sub_2618A70(_BYTE *a1)
{
  _BYTE *v1; // rax
  unsigned int v2; // r8d

  v1 = (_BYTE *)*((_QWORD *)a1 + 3);
  v2 = 0;
  if ( *v1 == 85 )
    LOBYTE(v2) = a1 == v1 - 32;
  return v2;
}
