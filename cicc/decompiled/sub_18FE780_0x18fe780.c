// Function: sub_18FE780
// Address: 0x18fe780
//
unsigned __int64 __fastcall sub_18FE780(__int64 a1)
{
  __int64 v2; // rax
  __int64 *v3; // rdi
  __int64 *v4; // rsi
  int v6; // [rsp+4h] [rbp-1Ch] BYREF
  unsigned __int64 v7; // [rsp+8h] [rbp-18h] BYREF

  v2 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v3 = *(__int64 **)(a1 - 8);
    v4 = &v3[v2];
  }
  else
  {
    v4 = (__int64 *)a1;
    v3 = (__int64 *)(a1 - v2 * 8);
  }
  v7 = sub_18FDB50(v3, v4);
  v6 = *(unsigned __int8 *)(a1 + 16) - 24;
  return sub_18FDAA0(&v6, (__int64 *)&v7);
}
