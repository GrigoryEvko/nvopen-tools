// Function: sub_DC7EB0
// Address: 0xdc7eb0
//
_QWORD *__fastcall sub_DC7EB0(__int64 *a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  if ( *(_DWORD *)(a2 + 8) == 1 )
    return **(_QWORD ***)a2;
  else
    return sub_DC5AB0(a1, a2, a3, a4);
}
