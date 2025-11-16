// Function: sub_11ECD00
// Address: 0x11ecd00
//
__int64 __fastcall sub_11ECD00(__int64 a1, __int64 a2, __int64 *a3)
{
  char v4; // r8
  __int64 result; // rax
  _QWORD *v6; // rdi
  __int64 **v7; // r13
  unsigned __int64 v8; // r15
  unsigned int v9; // r14d
  unsigned int v10; // eax
  unsigned __int64 v11; // rax
  unsigned __int8 *v12; // rax
  __int64 v13; // [rsp+18h] [rbp-68h]
  __int64 v14; // [rsp+20h] [rbp-60h] BYREF
  __int16 v15; // [rsp+40h] [rbp-40h]

  BYTE4(v14) = 0;
  BYTE4(v13) = 0;
  v4 = sub_11EC990(a1, a2, 3u, 0x100000002LL, v13, v14);
  result = 0;
  if ( v4 )
  {
    v6 = (_QWORD *)a3[9];
    v15 = 257;
    v7 = (__int64 **)sub_BCB2B0(v6);
    v8 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v9 = sub_BCB060(*(_QWORD *)(v8 + 8));
    v10 = sub_BCB060((__int64)v7);
    v11 = sub_11DB4B0(a3, (unsigned int)(v9 <= v10) + 38, v8, v7, (__int64)&v14, 0, v13, 0);
    v12 = (unsigned __int8 *)sub_B34240(
                               (__int64)a3,
                               *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
                               v11,
                               *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
                               0x100u,
                               0,
                               0,
                               0,
                               0);
    sub_11DAF00(v12, a2);
    return *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  }
  return result;
}
