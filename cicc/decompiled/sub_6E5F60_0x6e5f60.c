// Function: sub_6E5F60
// Address: 0x6e5f60
//
__int64 __fastcall sub_6E5F60(_DWORD *a1, FILE *a2, char a3)
{
  unsigned int v4; // r13d
  __int64 result; // rax

  v4 = sub_67F240();
  result = sub_6E53E0(a3, v4, a1);
  if ( (_DWORD)result )
    return sub_685A50(v4, a1, a2, a3);
  return result;
}
