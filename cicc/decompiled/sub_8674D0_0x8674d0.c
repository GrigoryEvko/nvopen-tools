// Function: sub_8674D0
// Address: 0x8674d0
//
__int64 __fastcall sub_8674D0(__int64 *a1, _QWORD *a2, unsigned int a3, __int64 a4, __int64 a5)
{
  char v5; // al
  __int64 result; // rax
  __int64 v7; // rax
  __int64 *v8; // rax

  if ( dword_4F04C44 == -1 && (v5 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6), (v5 & 2) == 0) )
  {
    if ( unk_4F04C48 == -1 )
      return sub_72C930();
    if ( (v5 & 6) != 0 )
      return sub_72C930();
    if ( !qword_4F04C18 )
      return sub_72C930();
    v7 = qword_4F04C18[2];
    if ( !v7 )
      return sub_72C930();
    v8 = *(__int64 **)(v7 + 8);
    if ( !v8 )
      return sub_72C930();
    while ( *((_DWORD *)v8 + 7) != dword_4F06650[0] )
    {
      v8 = (__int64 *)*v8;
      if ( !v8 )
        return sub_72C930();
    }
    result = *(_QWORD *)(v8[10] + 32);
    if ( !result )
      return sub_72C930();
  }
  else
  {
    sub_867130(*a1, a2, (__int64)a1, (_QWORD *)a3, a5);
    return (__int64)a1;
  }
  return result;
}
