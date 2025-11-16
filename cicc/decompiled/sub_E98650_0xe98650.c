// Function: sub_E98650
// Address: 0xe98650
//
unsigned __int64 __fastcall sub_E98650(_QWORD *a1, _DWORD *a2, unsigned __int64 a3, __int64 a4)
{
  unsigned int v5; // r8d
  __int64 v6; // rdx
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  void *v13; // rsi
  void *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // ecx
  __int64 v18; // [rsp+0h] [rbp-B0h] BYREF
  unsigned __int64 v19; // [rsp+8h] [rbp-A8h]
  _BYTE *v20[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v21; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v22[4]; // [rsp+30h] [rbp-80h] BYREF
  __int16 v23; // [rsp+50h] [rbp-60h]
  void *v24[4]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v25; // [rsp+80h] [rbp-30h]

  if ( a4 != *(_QWORD *)(a1[21] + 24LL) )
  {
    v5 = *(_DWORD *)(a4 + 152);
    if ( v5 == -1 )
    {
      v5 = (*a2)++;
      *(_DWORD *)(a4 + 152) = v5;
    }
    if ( (*(_BYTE *)(a4 + 149) & 0x10) == 0 )
    {
      v6 = 0;
      return sub_E6E2A0(a1, a3, v6, v5);
    }
    v6 = *(_QWORD *)(a4 + 160);
    if ( *(_BYTE *)(a1[19] + 19LL) )
      return sub_E6E2A0(a1, a3, v6, v5);
    v8 = *(_QWORD *)(a4 + 128);
    LOBYTE(v24[0]) = 36;
    v9 = *(_QWORD *)(a4 + 136);
    v18 = v8;
    v19 = v9;
    v10 = sub_C931B0(&v18, v24, 1u, 0);
    if ( v10 == -1 )
    {
      v13 = 0;
      v14 = 0;
    }
    else
    {
      v11 = v19;
      v12 = v10 + 1;
      v13 = 0;
      if ( v12 <= v19 )
      {
        v11 = v12;
        v13 = (void *)(v19 - v12);
      }
      v14 = (void *)(v18 + v11);
    }
    v24[3] = v13;
    v15 = *(_QWORD *)(a3 + 136);
    v16 = *(_QWORD *)(a3 + 128);
    v23 = 773;
    v22[1] = v15;
    v22[2] = "$";
    v22[0] = v16;
    v24[0] = v22;
    v24[2] = v14;
    v25 = 1282;
    sub_CA0F50((__int64 *)v20, v24);
    v17 = *(_DWORD *)(a3 + 148);
    BYTE1(v17) |= 0x10u;
    a3 = sub_E6DEB0(a1, v20[0], (size_t)v20[1], v17, (__int64)byte_3F871B3, 0, 2u, 0xFFFFFFFF);
    if ( (__int64 *)v20[0] != &v21 )
      j_j___libc_free_0(v20[0], v21 + 1);
  }
  return a3;
}
