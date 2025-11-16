// Function: sub_18E7070
// Address: 0x18e7070
//
void __fastcall sub_18E7070(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // r8
  int v10; // r9d
  char *v11; // [rsp+10h] [rbp-170h] BYREF
  __int64 v12; // [rsp+18h] [rbp-168h]
  _BYTE v13[128]; // [rsp+20h] [rbp-160h] BYREF
  __int64 v14; // [rsp+A0h] [rbp-E0h]
  int v15; // [rsp+A8h] [rbp-D8h]
  unsigned __int64 v16[2]; // [rsp+B0h] [rbp-D0h] BYREF
  _BYTE v17[128]; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v18; // [rsp+140h] [rbp-40h]
  int v19; // [rsp+148h] [rbp-38h]

  v7 = *(unsigned int *)(a3 + 8);
  v11 = v13;
  v12 = 0x800000000LL;
  if ( (_DWORD)v7 )
    sub_18E63F0((__int64)&v11, (char **)a3, v7, a4, a5, a6);
  v14 = *(_QWORD *)(a3 + 144);
  v15 = *(_DWORD *)(a3 + 152);
  sub_18E63F0(a3, (char **)a1, v7, a4, a5, a6);
  *(_QWORD *)(a3 + 144) = *(_QWORD *)(a1 + 144);
  *(_DWORD *)(a3 + 152) = *(_DWORD *)(a1 + 152);
  v16[1] = 0x800000000LL;
  v16[0] = (unsigned __int64)v17;
  if ( (_DWORD)v12 )
    sub_18E63F0((__int64)v16, &v11, v8, (__int64)v16, v9, v10);
  v18 = v14;
  v19 = v15;
  sub_18E6AF0(a1, 0, 0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 5), (unsigned __int64)v16, v9, v10);
  if ( (_BYTE *)v16[0] != v17 )
    _libc_free(v16[0]);
  if ( v11 != v13 )
    _libc_free((unsigned __int64)v11);
}
