// Function: sub_3753270
// Address: 0x3753270
//
_QWORD *__fastcall sub_3753270(_QWORD *a1, __int64 a2)
{
  char *v2; // rdx
  _QWORD *result; // rax
  char v4; // cl
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // rdx

  v2 = *(char **)(a2 + 8);
  result = a1;
  v4 = *v2;
  if ( *v2 == 17 )
  {
    v5 = *((_DWORD *)v2 + 8);
    if ( v5 <= 0x40 )
    {
      v6 = *((_QWORD *)v2 + 3);
      v7 = 0;
      if ( v5 )
        v7 = v6 << (64 - (unsigned __int8)v5) >> (64 - (unsigned __int8)v5);
      *result = 1;
      result[2] = 0;
      result[3] = v7;
    }
    else
    {
      *a1 = 2;
      a1[2] = 0;
      a1[3] = v2;
    }
  }
  else if ( v4 == 18 )
  {
    *a1 = 3;
    a1[2] = 0;
    a1[3] = v2;
  }
  else if ( v4 == 20 )
  {
    *a1 = 1;
    a1[2] = 0;
    a1[3] = 0;
  }
  else
  {
    a1[1] = 0;
    *a1 = 0x800000000LL;
    a1[2] = 0;
    a1[3] = 0;
    a1[4] = 0;
  }
  return result;
}
