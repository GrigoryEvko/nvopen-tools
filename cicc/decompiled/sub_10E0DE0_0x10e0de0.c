// Function: sub_10E0DE0
// Address: 0x10e0de0
//
__int64 __fastcall sub_10E0DE0(__int64 a1, unsigned int a2, char a3)
{
  __int64 v4; // rax
  unsigned int v6; // r14d
  __int64 v7; // r13
  __int64 v8; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v9; // [rsp+8h] [rbp-28h]

  v9 = a2;
  if ( a3 )
  {
    v4 = 0;
    if ( a2 > 0x40 )
    {
      sub_C43690((__int64)&v8, 0, 0);
      a2 = v9;
      v4 = v8;
    }
  }
  else
  {
    v6 = a2 - 1;
    v7 = 1LL << ((unsigned __int8)a2 - 1);
    if ( a2 > 0x40 )
    {
      sub_C43690((__int64)&v8, 0, 0);
      a2 = v9;
      if ( v9 > 0x40 )
      {
        *(_QWORD *)(v8 + 8LL * (v6 >> 6)) |= v7;
        a2 = v9;
        v4 = v8;
        goto LABEL_4;
      }
    }
    else
    {
      v8 = 0;
    }
    v4 = v7 | v8;
  }
LABEL_4:
  *(_QWORD *)a1 = v4;
  *(_BYTE *)(a1 + 12) = a3;
  *(_DWORD *)(a1 + 8) = a2;
  return a1;
}
