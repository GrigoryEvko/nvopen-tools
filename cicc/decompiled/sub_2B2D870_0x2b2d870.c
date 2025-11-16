// Function: sub_2B2D870
// Address: 0x2b2d870
//
__int64 __fastcall sub_2B2D870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbp
  int v7; // eax
  unsigned int v8; // r8d
  _BYTE *v9; // rax
  unsigned int v11; // eax
  __int64 v12; // rdi
  int *v14; // [rsp-58h] [rbp-58h] BYREF
  __int64 v15; // [rsp-50h] [rbp-50h]
  _QWORD v16[9]; // [rsp-48h] [rbp-48h] BYREF

  v7 = *(_DWORD *)(a2 + 104);
  if ( (unsigned int)(v7 - 1) <= 1 )
    return 3;
  v8 = 0;
  if ( v7 )
    return v8;
  v9 = *(_BYTE **)(a2 + 416);
  if ( *v9 != 61 )
    return v8;
  if ( v9 != *(_BYTE **)(a2 + 424) )
    return v8;
  v11 = *(_DWORD *)(a2 + 152);
  v8 = 1;
  if ( !v11 )
    return v8;
  v16[8] = v6;
  v12 = *(_QWORD *)(a2 + 144);
  v15 = 0xC00000000LL;
  v14 = (int *)v16;
  sub_2B0FC00(v12, v11, (__int64)&v14, 0xC00000000LL, 1, a6);
  if ( (unsigned __int8)sub_B4EDA0(v14, (unsigned int)v15, v15) )
  {
    if ( v14 != (int *)v16 )
      _libc_free((unsigned __int64)v14);
    return 5;
  }
  else
  {
    if ( v14 != (int *)v16 )
      _libc_free((unsigned __int64)v14);
    return 0;
  }
}
