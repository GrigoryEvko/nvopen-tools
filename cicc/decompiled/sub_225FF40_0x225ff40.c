// Function: sub_225FF40
// Address: 0x225ff40
//
__int64 __fastcall sub_225FF40(volatile signed __int32 ***a1)
{
  volatile signed __int32 *v1; // r12
  volatile signed __int32 *v2; // rbx
  __int64 result; // rax
  int v4; // edx

  v1 = **a1;
  if ( !v1 )
    sub_42641C(3u);
  v2 = v1 + 4;
  result = (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v1 + 16LL))(**a1);
  v4 = v1[4] & 0x7FFFFFFF;
  if ( v4 != 1 )
  {
    do
    {
      _InterlockedOr(v2, 0x80000000);
      result = sub_222D0C0((__int64)(v1 + 4), (__int64)(v1 + 4), v4 | 0x80000000, 0, 0, 0);
      v4 = *v2 & 0x7FFFFFFF;
    }
    while ( v4 != 1 && (_BYTE)result == 1 );
  }
  return result;
}
