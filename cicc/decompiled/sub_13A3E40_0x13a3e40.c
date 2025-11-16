// Function: sub_13A3E40
// Address: 0x13a3e40
//
__int64 __fastcall sub_13A3E40(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  unsigned __int64 v4; // rdi
  __int64 v5; // rdx
  unsigned __int64 v6; // rsi
  unsigned __int64 v8; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v9; // [rsp+8h] [rbp-18h]

  v3 = *(_DWORD *)(a2 + 8);
  v4 = *(_QWORD *)a2;
  v5 = 1LL << ((unsigned __int8)v3 - 1);
  if ( v3 > 0x40 )
  {
    if ( (*(_QWORD *)(v4 + 8LL * ((v3 - 1) >> 6)) & v5) != 0 )
    {
      v9 = *(_DWORD *)(a2 + 8);
      sub_16A4FD0(&v8, a2);
      LOBYTE(v3) = v9;
      if ( v9 > 0x40 )
      {
        sub_16A8F40(&v8);
        goto LABEL_5;
      }
      v6 = v8;
LABEL_4:
      v8 = ~v6 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v3);
LABEL_5:
      sub_16A7400(&v8);
      *(_DWORD *)(a1 + 8) = v9;
      *(_QWORD *)a1 = v8;
      return a1;
    }
    *(_DWORD *)(a1 + 8) = v3;
    sub_16A4FD0(a1, a2);
    return a1;
  }
  else
  {
    v6 = *(_QWORD *)a2;
    if ( (v5 & v4) != 0 )
    {
      v9 = v3;
      goto LABEL_4;
    }
    *(_DWORD *)(a1 + 8) = v3;
    *(_QWORD *)a1 = v4;
    return a1;
  }
}
