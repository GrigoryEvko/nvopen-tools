// Function: sub_C46CE0
// Address: 0xc46ce0
//
__int64 __fastcall sub_C46CE0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  __int64 v5; // rsi
  unsigned int v6; // r14d
  __int64 v7; // r13
  __int64 v8; // r13
  unsigned __int64 v9; // rax
  char v10; // [rsp+Fh] [rbp-31h] BYREF
  __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-28h]

  sub_C46BD0((__int64)&v11, a2, a3, &v10);
  if ( !v10 )
  {
    *(_DWORD *)(a1 + 8) = v12;
    *(_QWORD *)a1 = v11;
    return a1;
  }
  v4 = *(_DWORD *)(a2 + 8);
  v5 = *(_QWORD *)a2;
  v6 = v4 - 1;
  v7 = 1LL << ((unsigned __int8)v4 - 1);
  if ( v4 > 0x40 )
  {
    if ( (*(_QWORD *)(v5 + 8LL * (v6 >> 6)) & v7) != 0 )
    {
      *(_DWORD *)(a1 + 8) = v4;
      sub_C43690(a1, 0, 0);
      if ( *(_DWORD *)(a1 + 8) > 0x40u )
      {
        *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v6 >> 6)) |= v7;
        goto LABEL_11;
      }
      goto LABEL_7;
    }
    *(_DWORD *)(a1 + 8) = v4;
    v8 = ~v7;
    sub_C43690(a1, -1, 1);
    v9 = *(_QWORD *)a1;
    if ( *(_DWORD *)(a1 + 8) > 0x40u )
    {
LABEL_15:
      *(_QWORD *)(v9 + 8LL * (v6 >> 6)) &= v8;
      goto LABEL_11;
    }
LABEL_10:
    *(_QWORD *)a1 = v8 & v9;
    goto LABEL_11;
  }
  if ( (v7 & v5) == 0 )
  {
    *(_DWORD *)(a1 + 8) = v4;
    v8 = ~v7;
    *(_QWORD *)a1 = -1;
    sub_C43640((unsigned __int64 *)a1);
    v9 = *(_QWORD *)a1;
    if ( *(_DWORD *)(a1 + 8) > 0x40u )
      goto LABEL_15;
    goto LABEL_10;
  }
  *(_DWORD *)(a1 + 8) = v4;
  *(_QWORD *)a1 = 0;
LABEL_7:
  *(_QWORD *)a1 |= v7;
LABEL_11:
  if ( v12 <= 0x40 || !v11 )
    return a1;
  j_j___libc_free_0_0(v11);
  return a1;
}
