// Function: sub_253A610
// Address: 0x253a610
//
__int64 __fastcall sub_253A610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v7; // rdx
  _BYTE *v9; // [rsp+8h] [rbp-28h] BYREF
  __int64 v10; // [rsp+10h] [rbp-20h]
  _BYTE v11[24]; // [rsp+18h] [rbp-18h] BYREF

  v6 = 1;
  v7 = *(unsigned int *)(a2 + 16);
  v9 = v11;
  v10 = 0;
  if ( !(_DWORD)v7 )
    return v6;
  sub_2538240((__int64)&v9, (char **)(a2 + 8), v7, a4, a5, a6);
  LOBYTE(v6) = (_DWORD)v10 == 0;
  if ( v9 == v11 )
    return v6;
  _libc_free((unsigned __int64)v9);
  return v6;
}
