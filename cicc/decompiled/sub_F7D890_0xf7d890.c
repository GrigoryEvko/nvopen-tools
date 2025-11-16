// Function: sub_F7D890
// Address: 0xf7d890
//
__int64 __fastcall sub_F7D890(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v8; // cl
  char v9; // cl
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 *v16; // r15
  __int64 *v17; // [rsp+8h] [rbp-38h]

  while ( 1 )
  {
    if ( (*(_DWORD *)(a3 + 4) & 0x7FFFFFF) == 0 )
      return 0;
    v8 = *(_BYTE *)a3;
    if ( *(_BYTE *)a3 == 84 || (unsigned int)v8 - 67 <= 0xC && v8 != 78 )
      return 0;
    v9 = *(_BYTE *)(a3 + 7) & 0x40;
    if ( a1[58] == a4 )
    {
      v11 = 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      if ( v9 )
      {
        v12 = *(_QWORD *)(a3 - 8);
        v13 = v12 + v11;
      }
      else
      {
        v13 = a3;
        v12 = a3 - v11;
      }
      v14 = sub_F79960(v12, v13, 1);
      v17 = v15;
      if ( v15 != (__int64 *)v14 )
        break;
    }
LABEL_6:
    if ( v9 )
    {
      a3 = **(_QWORD **)(a3 - 8);
      if ( *(_BYTE *)a3 <= 0x1Cu )
        return 0;
    }
    else
    {
      a3 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      if ( *(_BYTE *)a3 <= 0x1Cu )
        return 0;
    }
    if ( (unsigned __int8)sub_B46970((unsigned __int8 *)a3) )
      return 0;
    if ( a2 == a3 )
      return 1;
  }
  v16 = (__int64 *)v14;
  while ( 1 )
  {
    if ( *(_BYTE *)*v16 > 0x1Cu )
    {
      result = sub_B19DB0(*(_QWORD *)(*a1 + 40LL), *v16, a1[59]);
      if ( !(_BYTE)result )
        return result;
    }
    v16 += 4;
    if ( v17 == v16 )
    {
      v9 = *(_BYTE *)(a3 + 7) & 0x40;
      goto LABEL_6;
    }
  }
}
