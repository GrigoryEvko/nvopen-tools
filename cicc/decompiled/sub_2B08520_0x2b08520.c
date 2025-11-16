// Function: sub_2B08520
// Address: 0x2b08520
//
__int64 __fastcall sub_2B08520(char *a1)
{
  char v1; // al

  v1 = *a1;
  if ( (unsigned __int8)*a1 > 0x1Cu && (v1 == 62 || (unsigned __int8)(v1 - 82) <= 1u || v1 == 91) )
    return *(_QWORD *)(*((_QWORD *)a1 - 8) + 8LL);
  else
    return *((_QWORD *)a1 + 1);
}
