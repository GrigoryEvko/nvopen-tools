// Function: sub_5E48B0
// Address: 0x5e48b0
//
__int64 __fastcall sub_5E48B0(__int64 a1)
{
  __int64 i; // rax
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 **v4; // rbx

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = *(_QWORD *)(*(_QWORD *)i + 96LL);
  result = *(unsigned __int8 *)(v2 + 178);
  if ( (result & 0x10) == 0 )
  {
    *(_BYTE *)(v2 + 178) = result | 0x10;
    result = *(_QWORD *)(a1 + 168);
    v4 = *(__int64 ***)result;
    if ( *(_QWORD *)result )
    {
      do
      {
        if ( ((_BYTE)v4[12] & 1) != 0 )
          result = sub_5E48B0(v4[5]);
        v4 = (__int64 **)*v4;
      }
      while ( v4 );
    }
  }
  return result;
}
