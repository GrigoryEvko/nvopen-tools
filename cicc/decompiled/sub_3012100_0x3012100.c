// Function: sub_3012100
// Address: 0x3012100
//
__int64 __fastcall sub_3012100(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  char v3; // dl
  bool v4; // zf
  __int64 result; // rax
  __int64 v6; // rdx

  v2 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == a1 + 48 || !v2 || (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
    BUG();
  v3 = *(_BYTE *)(v2 - 24);
  if ( v3 == 34 )
    return 0;
  if ( v3 == 39 )
  {
    v4 = **(_QWORD **)(v2 - 32) == a2;
    result = 0;
    if ( v4 )
      return a1;
  }
  else
  {
    v6 = *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 - 20) & 0x7FFFFFF) - 24);
    result = 0;
    if ( a2 == *(_QWORD *)(v6 - 32) )
      return *(_QWORD *)(v6 + 40);
  }
  return result;
}
