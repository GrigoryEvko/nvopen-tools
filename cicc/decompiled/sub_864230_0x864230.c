// Function: sub_864230
// Address: 0x864230
//
__int64 *__fastcall sub_864230(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rdi

  result = (__int64 *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  v4 = 0;
  if ( (unsigned __int8)(*((_BYTE *)result + 4) - 3) <= 1u )
    v4 = *(_QWORD *)(result[23] + 32);
  if ( v4 != a1 || (_DWORD)a2 )
  {
    v5 = *(_QWORD *)(a1 + 40);
    if ( v5 && *(_BYTE *)(v5 + 28) == 3 )
    {
      v6 = *(_QWORD *)(v5 + 32);
      if ( v6 )
        sub_864230(v6, a2);
    }
    return sub_8602E0(4u, a1);
  }
  else
  {
    ++result[66];
  }
  return result;
}
