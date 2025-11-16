// Function: sub_7AF060
// Address: 0x7af060
//
__int64 *__fastcall sub_7AF060(int a1)
{
  __int64 *result; // rax
  __int64 *v3; // rbx
  const char *v4; // r14
  const char *v5; // rcx
  unsigned __int64 v6; // rdx
  unsigned __int8 v7; // di
  _QWORD *v8; // rax

  result = sub_7AEFF0((unsigned __int64)qword_4F06410);
  v3 = result;
  if ( a1 )
  {
    result = &qword_4F06408;
    v4 = (const char *)qword_4F06408;
  }
  else
  {
    v4 = qword_4F06410;
  }
  if ( v3 )
  {
    while ( 1 )
    {
      result = (__int64 *)v3[13];
      if ( !result )
        break;
      v5 = (const char *)result[1];
      if ( v5 > v4 )
        break;
      if ( v5 == qword_4F06410
        || a1
        && (!*result || (v6 = *(_QWORD *)(*result + 8), v6 > qword_4F06408) && v6 != qword_4F06408 + 1LL)
        && *(_BYTE *)(qword_4F06408 + 1LL) )
      {
        v7 = 5;
        if ( dword_4D04964 )
          v7 = unk_4F07471;
        sub_6868B0(v7, 0x681u, &dword_4F063F8, (__int64)v5, result[2]);
      }
      v8 = (_QWORD *)v3[13];
      v3[13] = *v8;
      *v8 = qword_4F064A0;
      qword_4F064A0 = v8;
    }
  }
  return result;
}
