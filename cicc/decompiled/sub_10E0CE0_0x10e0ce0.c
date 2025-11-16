// Function: sub_10E0CE0
// Address: 0x10e0ce0
//
__int64 __fastcall sub_10E0CE0(__int64 a1, unsigned int a2, char a3)
{
  unsigned __int64 v4; // rax
  unsigned int v6; // r14d
  __int64 v7; // rbx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-28h]

  v10 = a2;
  if ( a3 )
  {
    if ( a2 > 0x40 )
    {
      sub_C43690((__int64)&v9, -1, 1);
      a2 = v10;
      v4 = v9;
    }
    else
    {
      v4 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a2;
      if ( !a2 )
        v4 = 0;
    }
  }
  else
  {
    v6 = a2 - 1;
    v7 = ~(1LL << ((unsigned __int8)a2 - 1));
    if ( a2 > 0x40 )
    {
      sub_C43690((__int64)&v9, -1, 1);
      a2 = v10;
      if ( v10 > 0x40 )
      {
        *(_QWORD *)(v9 + 8LL * (v6 >> 6)) &= v7;
        a2 = v10;
        v4 = v9;
        goto LABEL_5;
      }
    }
    else
    {
      v8 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a2;
      if ( !a2 )
        v8 = 0;
      v9 = v8;
    }
    v4 = v7 & v9;
  }
LABEL_5:
  *(_QWORD *)a1 = v4;
  *(_BYTE *)(a1 + 12) = a3;
  *(_DWORD *)(a1 + 8) = a2;
  return a1;
}
