// Function: sub_22A6980
// Address: 0x22a6980
//
void __fastcall sub_22A6980(__int64 a1, __int64 a2, char a3, int a4, char a5, char a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  _DWORD *v8; // rax
  _DWORD *v9; // rax
  __int64 *v10; // rax
  bool v11; // r8
  int v12; // eax
  __int64 v13; // rax

  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 8) = a5;
  *(_BYTE *)(a1 + 9) = a6;
  if ( a4 )
  {
    *(_BYTE *)(a1 + 10) = a3;
    *(_DWORD *)(a1 + 12) = a4;
    return;
  }
  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(a2 + 24);
  if ( v6 == 12 )
  {
    if ( *(_QWORD *)v7 == 0x75427761522E7864LL && *(_DWORD *)(v7 + 8) == 1919247974 )
    {
      *(_BYTE *)(a1 + 10) = **(_DWORD **)(a2 + 40) != 0;
      v10 = *(__int64 **)(a2 + 16);
      if ( *(_BYTE *)(*v10 + 8) == 7 || (v11 = sub_BCAC40(*v10, 8), v12 = 12, v11) )
        v12 = 11;
      goto LABEL_17;
    }
    if ( *(_QWORD *)v7 == 0x786554534D2E7864LL && *(_DWORD *)(v7 + 8) == 1701999988 )
    {
LABEL_7:
      v8 = *(_DWORD **)(a2 + 40);
      *(_BYTE *)(a1 + 10) = *v8 != 0;
      *(_DWORD *)(a1 + 12) = v8[3];
      return;
    }
LABEL_32:
    BUG();
  }
  if ( v6 == 14 )
  {
    if ( *(_QWORD *)v7 == 0x64657079542E7864LL && *(_DWORD *)(v7 + 8) == 1717990722 && *(_WORD *)(v7 + 12) == 29285 )
    {
      v9 = *(_DWORD **)(a2 + 40);
      *(_DWORD *)(a1 + 12) = 10;
      *(_BYTE *)(a1 + 10) = *v9 != 0;
      return;
    }
    goto LABEL_32;
  }
  if ( v6 != 10 )
  {
    if ( v6 == 18
      && !(*(_QWORD *)v7 ^ 0x62646565462E7864LL | *(_QWORD *)(v7 + 8) ^ 0x75747865546B6361LL)
      && *(_WORD *)(v7 + 16) == 25970 )
    {
      v13 = *(_QWORD *)(a2 + 40);
      *(_BYTE *)(a1 + 10) = 1;
      v12 = *(_DWORD *)(v13 + 4);
LABEL_17:
      *(_DWORD *)(a1 + 12) = v12;
      return;
    }
    goto LABEL_32;
  }
  if ( *(_QWORD *)v7 == 0x75747865542E7864LL && *(_WORD *)(v7 + 8) == 25970 )
    goto LABEL_7;
  if ( *(_QWORD *)v7 == 0x66667542432E7864LL && *(_WORD *)(v7 + 8) == 29285 )
  {
    *(_BYTE *)(a1 + 10) = 2;
    *(_DWORD *)(a1 + 12) = 13;
  }
  else
  {
    if ( *(_QWORD *)v7 != 0x6C706D61532E7864LL || *(_WORD *)(v7 + 8) != 29285 )
      goto LABEL_32;
    *(_BYTE *)(a1 + 10) = 3;
    *(_DWORD *)(a1 + 12) = 14;
  }
}
