// Function: sub_3960EF0
// Address: 0x3960ef0
//
__int64 __fastcall sub_3960EF0(_BYTE *a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // rdx
  __int64 v3; // rcx

  result = 0;
  if ( a1[16] == 17 )
  {
    v2 = *(unsigned __int8 *)(*(_QWORD *)a1 + 8LL);
    if ( (unsigned __int8)v2 <= 0x10u )
    {
      v3 = 100990;
      if ( _bittest64(&v3, v2) )
        return (unsigned int)sub_15E0420((__int64)a1, 6) ^ 1;
    }
  }
  return result;
}
