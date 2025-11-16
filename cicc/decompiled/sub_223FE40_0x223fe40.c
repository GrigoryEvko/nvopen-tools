// Function: sub_223FE40
// Address: 0x223fe40
//
__int64 __fastcall sub_223FE40(__int64 a1, unsigned int a2)
{
  __int64 v3; // r12
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 v6; // rsi
  _BYTE *v7; // rcx
  char v8; // r14
  unsigned __int64 v9; // rsi
  __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // r12
  unsigned __int64 v13; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-60h]
  _QWORD *v18; // [rsp+8h] [rbp-58h] BYREF
  __int64 v19; // [rsp+10h] [rbp-50h]
  _QWORD v20[9]; // [rsp+18h] [rbp-48h] BYREF

  if ( (*(_BYTE *)(a1 + 64) & 0x10) == 0 )
    return 0xFFFFFFFFLL;
  if ( a2 == -1 )
    return 0;
  v3 = *(_QWORD *)(a1 + 72);
  if ( v3 == a1 + 88 )
  {
    v5 = *(_QWORD *)(a1 + 32);
    v7 = *(_BYTE **)(a1 + 40);
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 48) - v5) > 0xE )
    {
      v8 = a2;
      v9 = 512;
      if ( *(_QWORD *)(a1 + 48) <= (unsigned __int64)v7 )
      {
LABEL_12:
        LOBYTE(v20[0]) = 0;
        v18 = v20;
        v19 = 0;
        sub_2240E30(&v18, v9);
        v10 = *(_QWORD *)(a1 + 32);
        if ( v10 )
          sub_2241130(&v18, 0, v19, v10, *(_QWORD *)(a1 + 48) - v10);
        v11 = (unsigned __int64)v18;
        v12 = v19;
        v13 = 15;
        if ( v18 != v20 )
          v13 = v20[0];
        v17 = v19 + 1;
        if ( v19 + 1 > v13 )
        {
          sub_2240BB0(&v18, v19, 0, 0, 1);
          v11 = (unsigned __int64)v18;
        }
        *(_BYTE *)(v11 + v12) = v8;
        v19 = v17;
        *((_BYTE *)v18 + v12 + 1) = 0;
        sub_22415E0(a1 + 72, &v18);
        sub_223FD50(
          a1,
          *(_QWORD *)(a1 + 72),
          *(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8),
          *(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32));
        if ( v18 != v20 )
          j___libc_free_0((unsigned __int64)v18);
        goto LABEL_20;
      }
LABEL_23:
      *v7 = v8;
LABEL_20:
      ++*(_QWORD *)(a1 + 40);
      return a2;
    }
    v6 = 15;
    goto LABEL_25;
  }
  v4 = *(_QWORD *)(a1 + 48);
  v5 = *(_QWORD *)(a1 + 32);
  v6 = *(_QWORD *)(a1 + 88);
  v7 = *(_BYTE **)(a1 + 40);
  if ( v4 - v5 < v6 )
  {
LABEL_25:
    sub_223FAE0((_QWORD *)a1, v3, v3 + v6, (__int64)&v7[-v5]);
    if ( (*(_BYTE *)(a1 + 64) & 8) != 0 )
    {
      v15 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 24) = v3 + *(_QWORD *)(a1 + 24) - *(_QWORD *)(a1 + 8) + 1;
      v16 = v3 + v15 - *(_QWORD *)(a1 + 8);
      *(_QWORD *)(a1 + 8) = v3;
      *(_QWORD *)(a1 + 16) = v16;
    }
    *(_BYTE *)(*(_QWORD *)(a1 + 40))++ = a2;
    return a2;
  }
  if ( (unsigned __int64)v7 < v4 || v6 != 0x3FFFFFFFFFFFFFFFLL )
  {
    v8 = a2;
    if ( (unsigned __int64)v7 >= v4 )
    {
      v9 = 2 * v6;
      if ( v9 > 0x3FFFFFFFFFFFFFFFLL )
        v9 = 0x3FFFFFFFFFFFFFFFLL;
      if ( v9 < 0x200 )
        v9 = 512;
      goto LABEL_12;
    }
    goto LABEL_23;
  }
  return 0xFFFFFFFFLL;
}
