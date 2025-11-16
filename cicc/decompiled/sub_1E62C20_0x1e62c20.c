// Function: sub_1E62C20
// Address: 0x1e62c20
//
__int64 __fastcall sub_1E62C20(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // r13
  __int64 *v3; // r12
  __int64 *v4; // r15
  __int64 v5; // r14
  __int64 result; // rax
  __int64 *v7; // r13
  __int64 *i; // r12
  unsigned __int64 v9; // [rsp+0h] [rbp-40h]

  if ( !(unsigned __int8)sub_1E62AD0(a1, (__int64)a2) )
    sub_16BD130("Broken region found: enumerated BB not in region!", 1u);
  v2 = a1[4];
  v9 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = (__int64 *)a2[12];
  if ( v3 != (__int64 *)a2[11] )
  {
    v4 = (__int64 *)a2[11];
    do
    {
      v5 = *v4;
      if ( (unsigned __int8)sub_1E62AD0(a1, *v4) != 1 && v2 != v5 )
        sub_16BD130("Broken region found: edges leaving the region must go to the exit node!", 1u);
      ++v4;
    }
    while ( v3 != v4 );
  }
  result = (__int64)a2;
  if ( a2 != (_QWORD *)v9 )
  {
    v7 = (__int64 *)a2[9];
    for ( i = (__int64 *)a2[8]; v7 != i; ++i )
    {
      result = sub_1E62AD0(a1, *i);
      if ( !(_BYTE)result )
        sub_16BD130("Broken region found: edges entering the region must go to the entry node!", 1u);
    }
  }
  return result;
}
