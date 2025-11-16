// Function: sub_866B90
// Address: 0x866b90
//
__int64 *__fastcall sub_866B90(__int64 a1)
{
  __int64 *result; // rax
  _QWORD *i; // rcx
  __int64 **v3; // rdx

  result = *(__int64 **)(*(_QWORD *)(a1 + 8) + 24LL);
  for ( i = *(_QWORD **)(*(_QWORD *)(a1 + 16) + 8LL); result; i = (_QWORD *)*i )
  {
    if ( !*((_DWORD *)result + 8) )
    {
      v3 = (__int64 **)i[10];
      if ( v3 )
      {
        v3 = (__int64 **)*v3;
        if ( v3 )
        {
          if ( ((_BYTE)v3[3] & 8) == 0 )
            v3 = 0;
        }
      }
      i[10] = v3;
    }
    result = (__int64 *)*result;
  }
  return result;
}
