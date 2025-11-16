// Function: sub_62F580
// Address: 0x62f580
//
_QWORD *sub_62F580()
{
  __int64 v0; // rax
  _QWORD *v1; // rbx
  _DWORD *v2; // rax
  _DWORD *v3; // rcx
  _DWORD *v4; // rdx
  _QWORD *result; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // rcx
  _QWORD *v8; // rdx

  v0 = sub_823970(16);
  qword_4CFDE38 = v0;
  if ( v0 )
  {
    v1 = (_QWORD *)v0;
    v2 = (_DWORD *)sub_823970(0x4000);
    v3 = v2;
    v4 = v2 + 4096;
    do
    {
      if ( v2 )
        *v2 = 0;
      v2 += 4;
    }
    while ( v2 != v4 );
    *v1 = v3;
    v1[1] = 1023;
  }
  result = (_QWORD *)sub_823970(16);
  qword_4CFDE40 = (__int64)result;
  v6 = result;
  if ( result )
  {
    result = (_QWORD *)sub_823970(0x8000);
    v7 = result;
    v8 = result + 4096;
    do
    {
      if ( result )
        *result = 0;
      result += 4;
    }
    while ( v8 != result );
    *v6 = v7;
    v6[1] = 1023;
  }
  return result;
}
