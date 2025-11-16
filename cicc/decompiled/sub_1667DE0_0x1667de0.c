// Function: sub_1667DE0
// Address: 0x1667de0
//
void __fastcall sub_1667DE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  char v3; // al
  unsigned int v4; // eax
  const char *v5; // rax
  __int64 v6; // r14
  _BYTE *v7; // rax
  __int64 v8; // rax
  const char *v9; // [rsp+0h] [rbp-40h] BYREF
  char v10; // [rsp+10h] [rbp-30h]
  char v11; // [rsp+11h] [rbp-2Fh]

  v2 = **(_QWORD **)(a2 - 48);
  if ( **(_QWORD **)(a2 - 24) == v2 )
  {
    v3 = *(_BYTE *)(v2 + 8);
    if ( v3 == 16 )
      v3 = *(_BYTE *)(**(_QWORD **)(v2 + 16) + 8LL);
    if ( (unsigned __int8)(v3 - 1) > 5u )
    {
      v11 = 1;
      v5 = "Invalid operand types for FCmp instruction";
    }
    else
    {
      v4 = *(unsigned __int16 *)(a2 + 18);
      BYTE1(v4) &= ~0x80u;
      if ( v4 <= 0xF )
      {
        sub_1663F80(a1, a2);
        return;
      }
      v11 = 1;
      v5 = "Invalid predicate in FCmp instruction!";
    }
  }
  else
  {
    v11 = 1;
    v5 = "Both operands to FCmp instruction are not of the same type!";
  }
  v6 = *(_QWORD *)a1;
  v9 = v5;
  v10 = 3;
  if ( v6 )
  {
    sub_16E2CE0(&v9, v6);
    v7 = *(_BYTE **)(v6 + 24);
    if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 16) )
    {
      sub_16E7DE0(v6, 10);
    }
    else
    {
      *(_QWORD *)(v6 + 24) = v7 + 1;
      *v7 = 10;
    }
    v8 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 72) = 1;
    if ( v8 )
      sub_164FA80((__int64 *)a1, a2);
  }
  else
  {
    *(_BYTE *)(a1 + 72) = 1;
  }
}
