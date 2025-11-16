// Function: sub_2433C90
// Address: 0x2433c90
//
__int64 __fastcall sub_2433C90(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned int v6; // r14d
  __int64 *v8; // rdi

  v4 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
  {
    v4 = **(_QWORD **)(v4 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
      v4 = **(_QWORD **)(v4 + 16);
  }
  if ( *(_DWORD *)(v4 + 8) >> 8 )
    return 1;
  v6 = sub_BD6020(a3);
  if ( (_BYTE)v6 )
    return 1;
  if ( sub_98C100(a3, 0) && (!*(_BYTE *)(a1 + 167) || (v8 = *(__int64 **)(a1 + 16)) != 0 && sub_D904B0(v8, a2)) )
  {
    return 1;
  }
  else if ( *sub_98ACB0((unsigned __int8 *)a3, 6u) == 3 )
  {
    return *(unsigned __int8 *)(a1 + 168) ^ 1u;
  }
  return v6;
}
