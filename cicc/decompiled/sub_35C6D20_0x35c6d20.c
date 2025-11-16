// Function: sub_35C6D20
// Address: 0x35c6d20
//
__int64 __fastcall sub_35C6D20(_QWORD *a1, __int64 a2)
{
  _DWORD *v2; // r15
  __int64 result; // rax
  __int64 v4; // rbx

  v2 = (_DWORD *)a1[4];
  result = (unsigned int)v2[16];
  if ( (_DWORD)result )
  {
    v4 = a1[41];
    if ( (_QWORD *)v4 != a1 + 40 )
    {
      do
      {
        if ( v4 + 48 != (*(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL)
          && sub_35C6A20(v2, a2, v4)
          && sub_35C6A20(v2, a2, v4) )
        {
          sub_C64ED0("Incomplete scavenging after 2nd pass", 1u);
        }
        v4 = *(_QWORD *)(v4 + 8);
      }
      while ( a1 + 40 != (_QWORD *)v4 );
    }
    result = sub_2EBEAA0((__int64)v2);
  }
  a1[43] |= 8uLL;
  return result;
}
