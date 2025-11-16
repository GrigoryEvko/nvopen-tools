// Function: sub_3205680
// Address: 0x3205680
//
__int64 __fastcall sub_3205680(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, char *a5, __int64 a6)
{
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // eax
  _BYTE *v16; // [rsp+0h] [rbp-90h] BYREF
  __int64 v17; // [rsp+8h] [rbp-88h]
  _BYTE v18[128]; // [rsp+10h] [rbp-80h] BYREF

  ++*(_DWORD *)(a2 + 1328);
  v16 = v18;
  v17 = 0x500000000LL;
  sub_31F7970(a2, a3, (__int64)&v16, (__int64)a4, (__int64)a5, a6);
  v9 = (__int64)v16;
  sub_31F5640(a1, (__int64)v16, (unsigned int)v17, a4, a5);
  if ( v16 != v18 )
    _libc_free((unsigned __int64)v16);
  v14 = *(_DWORD *)(a2 + 1328);
  if ( v14 == 1 )
  {
    sub_32053F0(a2, v9, v10, v11, v12, v13);
    v14 = *(_DWORD *)(a2 + 1328);
  }
  *(_DWORD *)(a2 + 1328) = v14 - 1;
  return a1;
}
