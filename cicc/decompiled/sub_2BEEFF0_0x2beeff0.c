// Function: sub_2BEEFF0
// Address: 0x2beeff0
//
__int64 *__fastcall sub_2BEEFF0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax

  if ( a3 != 15 )
  {
    if ( a3 == 16 )
    {
      if ( !(*(_QWORD *)a2 ^ 0x2D736E6F69676572LL | *(_QWORD *)(a2 + 8) ^ 0x7362622D6D6F7266LL) )
      {
        v7 = sub_22077B0(0x90u);
        v8 = v7;
        if ( v7 )
          sub_31B6490(v7, a4, a5);
        goto LABEL_8;
      }
    }
    else if ( a3 == 21
           && !(*(_QWORD *)a2 ^ 0x2D736E6F69676572LL | *(_QWORD *)(a2 + 8) ^ 0x74656D2D6D6F7266LL)
           && *(_DWORD *)(a2 + 16) == 1952539745
           && *(_BYTE *)(a2 + 20) == 97 )
    {
      v9 = sub_22077B0(0x90u);
      v8 = v9;
      if ( v9 )
        sub_31B6AE0(v9, a4, a5);
      goto LABEL_8;
    }
LABEL_3:
    *a1 = 0;
    return a1;
  }
  if ( *(_QWORD *)a2 != 0x6C6F632D64656573LL
    || *(_DWORD *)(a2 + 8) != 1952671084
    || *(_WORD *)(a2 + 12) != 28521
    || *(_BYTE *)(a2 + 14) != 110 )
  {
    goto LABEL_3;
  }
  v10 = sub_22077B0(0x90u);
  v8 = v10;
  if ( v10 )
    sub_31B7B20(v10, a4, a5);
LABEL_8:
  *a1 = v8;
  return a1;
}
