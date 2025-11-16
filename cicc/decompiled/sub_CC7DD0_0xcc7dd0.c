// Function: sub_CC7DD0
// Address: 0xcc7dd0
//
unsigned __int64 __fastcall sub_CC7DD0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  __int64 v3; // rsi
  unsigned __int64 v4; // r8
  unsigned __int8 v5; // dl
  unsigned int v6; // r10d
  __int64 v7; // rdi
  __int64 v8; // r9
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int128 v12; // [rsp+0h] [rbp-20h] BYREF

  if ( *(_DWORD *)(a1 + 32) != 11 || *(_DWORD *)(a1 + 44) != 38 )
    BUG();
  v1 = sub_CC7280((__int64 *)a1);
  if ( !*(_DWORD *)(a1 + 36) )
  {
    v10 = sub_CC7380((__int64 *)a1);
    v1 = (__int64)sub_CC5D50(v10, v11);
  }
  v3 = v1;
  v4 = v2;
  if ( v2 > 4 && *(_DWORD *)v1 == 1818851428 && *(_BYTE *)(v1 + 4) == 118 )
  {
    v4 = v2 - 5;
    v3 = v1 + 5;
  }
  v12 = 0;
  sub_F05080(&v12, v3, v4);
  if ( v12 < 0 )
  {
    v7 = (unsigned int)v12;
    v5 = 1;
    v8 = DWORD2(v12) & 0x7FFFFFFF;
    v6 = DWORD1(v12) & 0x7FFFFFFF;
  }
  else
  {
    v5 = BYTE7(v12) >> 7;
    v6 = DWORD1(v12) & 0x7FFFFFFF;
    v7 = (unsigned int)v12;
    v8 = DWORD2(v12) & 0x7FFFFFFF;
  }
  return v7 | (((v8 << 32) | v6 | ((unsigned __int64)v5 << 31)) << 32);
}
