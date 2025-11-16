// Function: sub_5EBA80
// Address: 0x5eba80
//
__int64 *__fastcall sub_5EBA80(__int64 **a1)
{
  __int64 *result; // rax
  __int64 *v2; // rdx
  __int64 v3; // rdx

  result = *a1;
  if ( *a1 )
  {
    v2 = (__int64 *)a1;
    while ( 1 )
    {
      v2[2] = 0;
      v2[1] = 0;
      v2 = result;
      if ( !*result )
        break;
      result = (__int64 *)*result;
    }
  }
  else
  {
    result = (__int64 *)a1;
  }
  v3 = qword_4CF7FB0;
  result[2] = 0;
  result[1] = 0;
  *result = v3;
  qword_4CF7FB0 = (__int64)a1;
  return result;
}
