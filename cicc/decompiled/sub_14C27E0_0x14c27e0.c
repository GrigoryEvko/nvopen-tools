// Function: sub_14C27E0
// Address: 0x14c27e0
//
__int64 __fastcall sub_14C27E0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // ebx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 16) != 13 )
  {
    if ( (unsigned __int8)sub_14C2730((__int64 *)a1, a2, a3, a4, a5, a6) )
      return sub_14BFF20(a1, a2, a3, a4, a5, a6);
    return 0;
  }
  v6 = *(_DWORD *)(a1 + 32);
  v7 = *(_QWORD *)(a1 + 24);
  v8 = 1LL << ((unsigned __int8)v6 - 1);
  if ( v6 > 0x40 )
    return (*(_QWORD *)(v7 + 8LL * ((v6 - 1) >> 6)) & v8) == 0 && v6 != (unsigned int)sub_16A57B0(a1 + 24);
  if ( (v8 & v7) != 0 )
    return 0;
  result = 1;
  if ( !v7 )
    return 0;
  return result;
}
