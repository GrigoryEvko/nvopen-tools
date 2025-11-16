// Function: sub_1EEB600
// Address: 0x1eeb600
//
_QWORD *__fastcall sub_1EEB600(_QWORD *a1, __int64 a2, __int64 a3, int a4, unsigned int a5, int a6)
{
  _DWORD *v6; // r15
  __int64 i; // rbx
  _QWORD *result; // rax

  v6 = (_DWORD *)a1[5];
  if ( v6[8] )
  {
    for ( i = a1[41]; a1 + 40 != (_QWORD *)i; i = *(_QWORD *)(i + 8) )
    {
      if ( i + 24 != (*(_QWORD *)(i + 24) & 0xFFFFFFFFFFFFFFF8LL)
        && sub_1EEB300(v6, a2, i, a4, a5, a6)
        && sub_1EEB300(v6, a2, i, a4, a5, a6) )
      {
        sub_16BD130("Incomplete scavenging after 2nd pass", 1u);
      }
    }
    sub_1E69990((__int64)v6);
  }
  result = (_QWORD *)a1[44];
  *result |= 8uLL;
  return result;
}
