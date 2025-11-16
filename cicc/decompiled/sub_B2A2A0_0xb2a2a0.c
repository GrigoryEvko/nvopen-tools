// Function: sub_B2A2A0
// Address: 0xb2a2a0
//
void __fastcall sub_B2A2A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rdi
  unsigned __int64 v6; // rdx
  char v7; // cl
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rcx
  char v17[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v18; // [rsp+8h] [rbp-38h]
  __int64 v19; // [rsp+10h] [rbp-30h]
  unsigned __int64 v20; // [rsp+18h] [rbp-28h]

  v4 = *(unsigned int *)(a2 + 624);
  if ( !*(_DWORD *)(a2 + 624) )
    return;
  if ( v4 == 1 )
  {
    v12 = sub_B22D80(a2);
    if ( a3 )
    {
      v14 = *(unsigned int *)(a3 + 624);
      v17[0] = 0;
      v15 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      v18 = a3;
      v20 = v14;
      v19 = a3;
      if ( (v13 & 4) != 0 )
        sub_B2A010(a1, (__int64)v17, v12, v15);
      else
        sub_B29E30(a1, (__int64)v17, v12, v15);
    }
    else
    {
      v16 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v13 & 4) != 0 )
        sub_B2A010(a1, 0, v12, v16);
      else
        sub_B29E30(a1, 0, v12, v16);
    }
    return;
  }
  v19 = a3;
  v6 = *(unsigned int *)(a1 + 56);
  v17[0] = 0;
  v18 = a2;
  v20 = v4;
  if ( v6 <= 0x64 )
  {
    v7 = 0;
    if ( v4 <= v6 )
      goto LABEL_5;
LABEL_13:
    sub_B28E70(a1, (__int64)v17);
    v7 = v17[0];
    if ( !v20 )
      return;
    goto LABEL_5;
  }
  v7 = 0;
  if ( v6 / 0x28 < v4 )
    goto LABEL_13;
LABEL_5:
  v8 = 0;
  while ( !v7 )
  {
    v9 = sub_B22D80(v18);
    v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v10 & 4) != 0 )
    {
      sub_B2A010(a1, (__int64)v17, v9, v11);
      if ( v20 <= ++v8 )
        return;
    }
    else
    {
      sub_B29E30(a1, (__int64)v17, v9, v11);
      if ( v20 <= ++v8 )
        return;
    }
    v7 = v17[0];
  }
}
