// Function: sub_2730CF0
// Address: 0x2730cf0
//
__int64 __fastcall sub_2730CF0(__int64 a1)
{
  _QWORD **v1; // rbx
  _QWORD **i; // r12
  __int64 result; // rax

  v1 = *(_QWORD ***)(a1 + 5608);
  for ( i = &v1[2 * *(unsigned int *)(a1 + 5616)]; i != v1; v1 += 2 )
  {
    while ( (*v1)[2] )
    {
      v1 += 2;
      if ( i == v1 )
        return result;
    }
    result = sub_B43D60(*v1);
  }
  return result;
}
