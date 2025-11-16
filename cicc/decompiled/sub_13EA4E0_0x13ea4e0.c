// Function: sub_13EA4E0
// Address: 0x13ea4e0
//
int *__fastcall sub_13EA4E0(int *a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v6; // rsi
  __int64 v7; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-38h]
  __int64 v9; // [rsp+10h] [rbp-30h]
  unsigned int v10; // [rsp+18h] [rbp-28h]

  v2 = (unsigned int)*(unsigned __int8 *)(a2 + 16) - 24;
  if ( (unsigned int)v2 <= 0x36
    && (v3 = 0x40000040000020LL, _bittest64(&v3, v2))
    && (*(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0)
    && (v6 = sub_1625790(a2, 4)) != 0
    && *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 11 )
  {
    sub_1593050(&v7, v6);
    sub_13EA060(a1, &v7);
    if ( v10 > 0x40 && v9 )
      j_j___libc_free_0_0(v9);
    if ( v8 > 0x40 && v7 )
      j_j___libc_free_0_0(v7);
  }
  else
  {
    *a1 = 4;
  }
  return a1;
}
