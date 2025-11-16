// Function: sub_22173C0
// Address: 0x22173c0
//
unsigned int *__fastcall sub_22173C0(__int64 a1, unsigned __int16 a2, unsigned int *a3, unsigned __int64 a4)
{
  unsigned int *i; // r12

  for ( i = a3; a4 > (unsigned __int64)i; ++i )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a1 + 16LL))(a1, a2, *i) )
      break;
  }
  return i;
}
