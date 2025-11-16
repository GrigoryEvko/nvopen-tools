// Function: sub_2F06B50
// Address: 0x2f06b50
//
__int64 __fastcall sub_2F06B50(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 i; // r14
  __int64 v8; // rdx
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 32) )
  {
    v6 = a2[6];
    for ( i = a2[7]; i != v6; result = sub_2F06990(a1, (__int64)a2, v8, a4, a5, a6) )
    {
      v8 = v6;
      v6 += 256;
    }
  }
  if ( a2[41] )
    return sub_2F06990(a1, (__int64)a2, (__int64)(a2 + 41), a4, a5, a6);
  return result;
}
