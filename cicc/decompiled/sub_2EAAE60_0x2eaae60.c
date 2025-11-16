// Function: sub_2EAAE60
// Address: 0x2eaae60
//
__int64 __fastcall sub_2EAAE60(__int64 a1, int a2)
{
  __int64 (*v2)(); // rax
  unsigned __int16 *v4; // rax
  __int64 v5; // rdx
  unsigned __int16 *v6; // rcx

  v2 = *(__int64 (**)())(*(_QWORD *)a1 + 1320LL);
  if ( v2 == sub_2EAAD00 )
    return 0;
  v4 = (unsigned __int16 *)v2();
  v5 *= 16;
  v6 = (unsigned __int16 *)((char *)v4 + v5);
  if ( (unsigned __int16 *)((char *)v4 + v5) == v4 )
    return 0;
  while ( *v4 != a2 )
  {
    v4 += 8;
    if ( v6 == v4 )
      return 0;
  }
  return *((_QWORD *)v4 + 1);
}
