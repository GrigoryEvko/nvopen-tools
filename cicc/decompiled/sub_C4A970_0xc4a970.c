// Function: sub_C4A970
// Address: 0xc4a970
//
__int64 __fastcall sub_C4A970(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // eax
  __int64 v7; // rdx
  unsigned int v8; // r15d
  __int64 v9; // r14
  unsigned int v10; // r8d
  bool v11; // di
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // r14
  bool v15; // [rsp+Fh] [rbp-41h] BYREF
  __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-38h]

  sub_C4A7C0((__int64)&v16, a2, a3, &v15);
  if ( !v15 )
  {
    *(_DWORD *)(a1 + 8) = v17;
    *(_QWORD *)a1 = v16;
    return a1;
  }
  v6 = *(_DWORD *)(a2 + 8);
  v7 = *(_QWORD *)a2;
  v8 = v6 - 1;
  v9 = 1LL << ((unsigned __int8)v6 - 1);
  if ( v6 > 0x40 )
    v7 = *(_QWORD *)(v7 + 8LL * (v8 >> 6));
  v10 = *(_DWORD *)(a3 + 8);
  v11 = (v7 & v9) != 0;
  v12 = *(_QWORD *)a3;
  if ( v10 > 0x40 )
    v12 = *(_QWORD *)(v12 + 8LL * ((v10 - 1) >> 6));
  *(_DWORD *)(a1 + 8) = v6;
  if ( ((v12 & (1LL << ((unsigned __int8)v10 - 1))) != 0) == v11 )
  {
    if ( v6 > 0x40 )
    {
      sub_C43690(a1, -1, 1);
    }
    else
    {
      *(_QWORD *)a1 = -1;
      sub_C43640((unsigned __int64 *)a1);
    }
    v13 = *(_QWORD *)a1;
    v14 = ~v9;
    if ( *(_DWORD *)(a1 + 8) > 0x40u )
      *(_QWORD *)(v13 + 8LL * (v8 >> 6)) &= v14;
    else
      *(_QWORD *)a1 = v13 & v14;
    goto LABEL_12;
  }
  if ( v6 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    if ( *(_DWORD *)(a1 + 8) > 0x40u )
    {
      *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v8 >> 6)) |= v9;
      goto LABEL_12;
    }
  }
  else
  {
    *(_QWORD *)a1 = 0;
  }
  *(_QWORD *)a1 |= v9;
LABEL_12:
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  return a1;
}
