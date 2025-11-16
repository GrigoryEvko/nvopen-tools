// Function: sub_16820A0
// Address: 0x16820a0
//
__int64 __fastcall sub_16820A0(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned int v5; // r8d
  _BYTE *v6; // rcx
  char *v7; // r9
  __int64 v8; // rcx
  _QWORD v9[4]; // [rsp-20h] [rbp-20h] BYREF

  result = 0;
  if ( a1 )
  {
    for ( v9[0] = sub_16875A0(); !(unsigned __int8)sub_1687600(v9); v9[0] = sub_1687670(v9[0]) )
    {
      v3 = sub_16880C0(v9[0]);
      v6 = a2;
      v7 = (char *)v3;
      do
      {
        if ( (unsigned __int8)sub_1682000(v7, v6, v4, (__int64)v6, v5) )
          return 1;
        if ( *v7 != 42 )
          break;
        v6 = (_BYTE *)(v8 + 1);
      }
      while ( *(v6 - 1) );
    }
    return 0;
  }
  return result;
}
