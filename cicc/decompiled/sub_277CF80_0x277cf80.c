// Function: sub_277CF80
// Address: 0x277cf80
//
unsigned __int64 __fastcall sub_277CF80(__int64 *a1)
{
  __int64 *v1; // r12
  __int64 v3; // rax
  __int64 *v4; // rdi
  __int64 v6; // rax
  __int64 *v7; // rdi
  int v8; // [rsp+Ch] [rbp-24h] BYREF
  unsigned __int64 v9; // [rsp+10h] [rbp-20h] BYREF
  __int64 v10[3]; // [rsp+18h] [rbp-18h] BYREF

  v1 = a1;
  if ( (unsigned __int8)sub_A73ED0(a1 + 9, 6) || (unsigned __int8)sub_B49560((__int64)a1, 6) )
  {
    v3 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
    if ( (*((_BYTE *)a1 + 7) & 0x40) != 0 )
    {
      v4 = (__int64 *)*(a1 - 1);
      v1 = &v4[(unsigned __int64)v3 / 8];
    }
    else
    {
      v4 = &a1[v3 / 0xFFFFFFFFFFFFFFF8LL];
    }
    v9 = sub_F58E90(v4, v1);
    v10[0] = a1[5];
    v8 = *(unsigned __int8 *)a1 - 29;
    return sub_277CBD0(&v8, v10, (__int64 *)&v9);
  }
  else
  {
    v6 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
    v7 = &a1[v6 / 0xFFFFFFFFFFFFFFF8LL];
    if ( (*((_BYTE *)a1 + 7) & 0x40) != 0 )
    {
      v7 = (__int64 *)*(a1 - 1);
      v1 = &v7[(unsigned __int64)v6 / 8];
    }
    v10[0] = sub_F58E90(v7, v1);
    LODWORD(v9) = *(unsigned __int8 *)a1 - 29;
    return sub_C4ECF0((int *)&v9, v10);
  }
}
