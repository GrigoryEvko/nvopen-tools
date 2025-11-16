// Function: sub_139F030
// Address: 0x139f030
//
__int64 __fastcall sub_139F030(__int64 a1)
{
  int v1; // edx
  unsigned __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rax

  v1 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned int)(v1 - 25) <= 9 )
    return 1;
  if ( (_BYTE)v1 == 78 )
  {
    v5 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v5 + 16) && (*(_BYTE *)(v5 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v5 + 36) - 35) <= 3 )
      return 1;
  }
  else
  {
    v3 = (unsigned int)(v1 - 34);
    if ( (unsigned int)v3 <= 0x36 )
    {
      v4 = 0x40018000000001LL;
      if ( _bittest64(&v4, v3) )
        return 1;
    }
  }
  if ( (unsigned __int8)sub_15F3040(a1) )
    return 1;
  else
    return sub_15F3330(a1);
}
