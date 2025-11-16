// Function: sub_C47700
// Address: 0xc47700
//
__int64 __fastcall sub_C47700(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned int i; // r13d
  unsigned int v7; // edi
  __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  unsigned int v10; // eax
  unsigned __int64 v12; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-38h]

  sub_C449B0(a1, (const void **)a3, a2);
  for ( i = *(_DWORD *)(a3 + 8); a2 > i; i *= 2 )
  {
    v10 = *(_DWORD *)(a1 + 8);
    v13 = v10;
    if ( v10 <= 0x40 )
    {
      v7 = v10;
      v12 = *(_QWORD *)a1;
    }
    else
    {
      sub_C43780((__int64)&v12, (const void **)a1);
      v10 = v13;
      if ( v13 > 0x40 )
      {
        sub_C47690((__int64 *)&v12, i);
        if ( *(_DWORD *)(a1 + 8) <= 0x40u )
          goto LABEL_9;
        goto LABEL_17;
      }
      v7 = *(_DWORD *)(a1 + 8);
    }
    v8 = 0;
    if ( i != v10 )
      v8 = v12 << i;
    v9 = v8 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v10);
    if ( !v10 )
      v9 = 0;
    v12 = v9;
    if ( v7 <= 0x40 )
    {
LABEL_9:
      *(_QWORD *)a1 |= v12;
      goto LABEL_10;
    }
LABEL_17:
    sub_C43BD0((_QWORD *)a1, (__int64 *)&v12);
LABEL_10:
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
  }
  return a1;
}
