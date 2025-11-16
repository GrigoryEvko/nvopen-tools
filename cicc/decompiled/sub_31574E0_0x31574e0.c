// Function: sub_31574E0
// Address: 0x31574e0
//
char __fastcall sub_31574E0(unsigned __int8 *a1)
{
  unsigned __int8 *v1; // rbx
  char result; // al
  int v3; // edx
  __int64 v4; // rax
  unsigned __int8 *v5; // r12

  v1 = a1;
  result = sub_AC30F0((__int64)a1);
  if ( !result )
  {
    v3 = *a1;
    if ( (unsigned int)(v3 - 12) <= 1 )
      return 1;
    if ( (unsigned int)(v3 - 9) <= 2 )
    {
      v4 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
      if ( (a1[7] & 0x40) != 0 )
      {
        v5 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
        v1 = &v5[v4];
      }
      else
      {
        v5 = &a1[-v4];
      }
      if ( v1 == v5 )
        return 1;
      while ( 1 )
      {
        result = sub_31574E0(*(_QWORD *)v5);
        if ( !result )
          break;
        v5 += 32;
        if ( v1 == v5 )
          return 1;
      }
    }
  }
  return result;
}
