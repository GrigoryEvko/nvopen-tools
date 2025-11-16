// Function: sub_D90F70
// Address: 0xd90f70
//
_QWORD *__fastcall sub_D90F70(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // rax
  _QWORD *v4; // rcx
  _QWORD *result; // rax
  unsigned int v6; // edx

  v2 = (a2 - (__int64)a1) >> 5;
  v3 = (a2 - (__int64)a1) >> 3;
  if ( v2 <= 0 )
  {
LABEL_11:
    switch ( v3 )
    {
      case 2LL:
        v6 = dword_4F88F88;
        result = a1;
        break;
      case 3LL:
        v6 = dword_4F88F88;
        result = a1;
        if ( *(unsigned __int16 *)(*a1 + 26LL) >= (unsigned int)dword_4F88F88 )
          return result;
        result = a1 + 1;
        break;
      case 1LL:
        v6 = dword_4F88F88;
LABEL_18:
        result = a1;
        if ( *(unsigned __int16 *)(*a1 + 26LL) < v6 )
          return (_QWORD *)a2;
        return result;
      default:
        return (_QWORD *)a2;
    }
    if ( *(unsigned __int16 *)(*result + 26LL) >= v6 )
      return result;
    a1 = result + 1;
    goto LABEL_18;
  }
  v4 = &a1[4 * v2];
  while ( 1 )
  {
    if ( *(unsigned __int16 *)(*a1 + 26LL) >= (unsigned int)dword_4F88F88 )
      return a1;
    if ( dword_4F88F88 <= (unsigned int)*(unsigned __int16 *)(a1[1] + 26LL) )
      return a1 + 1;
    if ( dword_4F88F88 <= (unsigned int)*(unsigned __int16 *)(a1[2] + 26LL) )
      return a1 + 2;
    if ( dword_4F88F88 <= (unsigned int)*(unsigned __int16 *)(a1[3] + 26LL) )
      return a1 + 3;
    a1 += 4;
    if ( a1 == v4 )
    {
      v3 = (a2 - (__int64)a1) >> 3;
      goto LABEL_11;
    }
  }
}
