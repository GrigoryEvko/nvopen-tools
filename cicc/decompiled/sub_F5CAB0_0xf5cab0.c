// Function: sub_F5CAB0
// Address: 0xf5cab0
//
__int64 __fastcall sub_F5CAB0(char *a1, __int64 *a2, _QWORD *a3, __int64 a4)
{
  if ( (unsigned __int8)*a1 <= 0x1Cu )
    return 0;
  if ( sub_F50EE0((unsigned __int8 *)a1, a2) )
    return sub_F5C810(a1, a2, a3, a4);
  return 0;
}
