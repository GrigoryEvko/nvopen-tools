// Function: sub_FFE760
// Address: 0xffe760
//
char __fastcall sub_FFE760(__int64 a1, __int64 a2, __int64 a3)
{
  char result; // al

  if ( *(_BYTE *)a1 <= 0x1Cu )
    return 1;
  if ( a3 )
    return sub_B19DB0(a3, a1, a2);
  result = sub_AA5B70(*(_QWORD *)(a1 + 40));
  if ( result )
    return *(_BYTE *)a1 != 40 && *(_BYTE *)a1 != 34;
  return result;
}
