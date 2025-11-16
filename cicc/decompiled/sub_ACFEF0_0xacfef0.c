// Function: sub_ACFEF0
// Address: 0xacfef0
//
__int64 __fastcall sub_ACFEF0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r15d
  char v3; // r14
  unsigned int v4; // r13d
  __int64 v6; // rbx
  _BYTE *v7; // rdi
  __int64 result; // rax
  __int64 v9; // rdx

  v2 = (unsigned __int8)a2;
  v3 = a2;
  v4 = a2;
  v6 = a1[2];
  if ( v6 )
  {
    while ( 1 )
    {
      v7 = *(_BYTE **)(v6 + 24);
      if ( *v7 > 0x15u )
        return 0;
      if ( *v7 <= 3u )
        return 0;
      a2 = v2;
      if ( !(unsigned __int8)sub_ACFEF0(v7, v2) )
        return 0;
      if ( v3 )
        v6 = a1[2];
      else
        v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        goto LABEL_10;
    }
  }
  else
  {
LABEL_10:
    result = 1;
    if ( (_BYTE)v4 )
    {
      sub_BA58F0(a1);
      sub_ACFDF0(a1, a2, v9);
      return v4;
    }
  }
  return result;
}
