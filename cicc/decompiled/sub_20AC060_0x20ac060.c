// Function: sub_20AC060
// Address: 0x20ac060
//
char __fastcall sub_20AC060(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  bool v6; // r14
  int v7; // eax
  __int64 v8; // rdi
  unsigned int v9; // ebx
  bool v10; // al
  char result; // al
  __int64 v12; // rdi
  unsigned int v13; // ebx
  __int64 v14; // rdi
  unsigned int v15; // r12d
  _QWORD v16[8]; // [rsp+0h] [rbp-40h] BYREF

  if ( (_BYTE)a3 == 2 )
  {
    v12 = *(_QWORD *)(a2 + 88);
    v13 = *(_DWORD *)(v12 + 32);
    if ( v13 <= 0x40 )
      return *(_QWORD *)(v12 + 24) == 1;
    else
      return v13 - 1 == (unsigned int)sub_16A57B0(v12 + 24);
  }
  v16[0] = a3;
  v16[1] = a4;
  if ( (_BYTE)a3 )
  {
    if ( (unsigned __int8)(a3 - 14) > 0x5Fu )
    {
      v6 = (unsigned __int8)(a3 - 86) <= 0x17u || (unsigned __int8)(a3 - 8) <= 5u;
      goto LABEL_5;
    }
  }
  else
  {
    v6 = sub_1F58CD0((__int64)v16);
    if ( !sub_1F58D20((__int64)v16) )
    {
LABEL_5:
      if ( v6 )
        v7 = a1[16];
      else
        v7 = a1[15];
      if ( v7 != 1 )
        goto LABEL_8;
LABEL_16:
      v14 = *(_QWORD *)(a2 + 88);
      v15 = *(_DWORD *)(v14 + 32);
      if ( v15 <= 0x40 )
      {
        if ( *(_QWORD *)(v14 + 24) == 1 )
        {
LABEL_18:
          result = 1;
          if ( !a5 )
            return result;
          return **(_BYTE **)(a2 + 40) != 2;
        }
      }
      else if ( (unsigned int)sub_16A57B0(v14 + 24) == v15 - 1 )
      {
        goto LABEL_18;
      }
      result = 0;
      if ( !a5 )
        return result;
      return **(_BYTE **)(a2 + 40) != 2;
    }
  }
  if ( a1[17] == 1 )
    goto LABEL_16;
LABEL_8:
  v8 = *(_QWORD *)(a2 + 88);
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 <= 0x40 )
    v10 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) == *(_QWORD *)(v8 + 24);
  else
    v10 = v9 == (unsigned int)sub_16A58F0(v8 + 24);
  return a5 & v10;
}
