// Function: sub_864360
// Address: 0x864360
//
__int64 *__fastcall sub_864360(__int64 a1, int a2)
{
  int v3; // r13d
  __int64 *result; // rax
  __int64 v5; // rdx
  char v6; // bl
  __int64 v7; // rax
  __int64 v8; // rdi

  v3 = dword_4F04C64;
  result = (__int64 *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  v5 = 0;
  v6 = *((_BYTE *)result + 4);
  if ( (unsigned __int8)(v6 - 3) <= 1u )
    v5 = *(_QWORD *)(result[23] + 32);
  if ( v5 != a1 || a2 )
  {
    v7 = *(_QWORD *)(a1 + 40);
    if ( v7 )
    {
      if ( *(_BYTE *)(v7 + 28) == 3 )
      {
        v8 = *(_QWORD *)(v7 + 32);
        if ( v8 )
          sub_864360(v8, 0);
      }
    }
    result = sub_8602E0(5u, a1);
    if ( v6 == 8 )
      return (__int64 *)sub_85B070(v3);
  }
  else
  {
    ++result[66];
  }
  return result;
}
