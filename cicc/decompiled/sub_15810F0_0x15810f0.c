// Function: sub_15810F0
// Address: 0x15810f0
//
__int64 __fastcall sub_15810F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // r8
  unsigned int v7; // ebx
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned int v14; // ebx
  unsigned int v15; // ebx

  if ( a1 == a2 )
    return 1;
  if ( *(_BYTE *)(a1 + 16) == 5 )
    return 16;
  if ( *(_BYTE *)(a2 + 16) == 5 )
  {
    v13 = sub_15810F0(a2, a1);
    if ( v13 != 16 )
      return sub_15FF5D0(v13);
    return 16;
  }
  v5 = sub_15A36D0(1, a1, a2, 0, a5);
  if ( *(_BYTE *)(v5 + 16) == 13 )
  {
    v7 = *(_DWORD *)(v5 + 32);
    if ( !(v7 <= 0x40 ? *(_QWORD *)(v5 + 24) == 0 : v7 == (unsigned int)sub_16A57B0(v5 + 24)) )
      return 1;
  }
  v10 = sub_15A36D0(4, a1, a2, 0, v6);
  if ( *(_BYTE *)(v10 + 16) != 13 )
    goto LABEL_11;
  v14 = *(_DWORD *)(v10 + 32);
  if ( v14 <= 0x40 )
  {
    if ( *(_QWORD *)(v10 + 24) )
      return 4;
    goto LABEL_11;
  }
  v11 = (unsigned int)sub_16A57B0(v10 + 24);
  result = 4;
  if ( v14 == (_DWORD)v11 )
  {
LABEL_11:
    v12 = sub_15A36D0(2, a1, a2, 0, v11);
    if ( *(_BYTE *)(v12 + 16) == 13 )
    {
      v15 = *(_DWORD *)(v12 + 32);
      if ( !(v15 <= 0x40 ? *(_QWORD *)(v12 + 24) == 0 : v15 == (unsigned int)sub_16A57B0(v12 + 24)) )
        return 2;
    }
    return 16;
  }
  return result;
}
