// Function: sub_11E3930
// Address: 0x11e3930
//
__int64 __fastcall sub_11E3930(__int64 a1, __int64 a2, __int64 *a3)
{
  int v4; // edx
  unsigned __int64 v5; // r8
  __int64 v6; // r13
  __int64 v7; // rax
  _QWORD *v8; // rdi
  __int64 **v9; // r14
  unsigned int v10; // eax
  unsigned __int64 v11; // rax
  unsigned __int8 *v12; // rax
  __int64 v13; // r8
  unsigned __int64 v15; // [rsp+10h] [rbp-80h]
  unsigned int v16; // [rsp+1Ch] [rbp-74h]
  int v17; // [rsp+28h] [rbp-68h]
  int v18[8]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v19; // [rsp+50h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 4);
  v5 = *(_QWORD *)(a1 + 16);
  v18[0] = 0;
  v6 = *(_QWORD *)(a2 + 32 * (2LL - (v4 & 0x7FFFFFF)));
  sub_11DAA90(a2, v18, 1, v6, v5);
  v7 = *(_QWORD *)(a2 - 32);
  if ( !v7
    || *(_BYTE *)v7
    || *(_QWORD *)(v7 + 24) != *(_QWORD *)(a2 + 80)
    || (v13 = 0, (*(_BYTE *)(v7 + 33) & 0x20) == 0) )
  {
    v8 = (_QWORD *)a3[9];
    v19 = 257;
    v9 = (__int64 **)sub_BCB2B0(v8);
    v15 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v16 = sub_BCB060(*(_QWORD *)(v15 + 8));
    v10 = sub_BCB060((__int64)v9);
    v11 = sub_11DB4B0(a3, (unsigned int)(v16 <= v10) + 38, v15, v9, (__int64)v18, 0, v17, 0);
    v12 = (unsigned __int8 *)sub_B34240(
                               (__int64)a3,
                               *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
                               v11,
                               v6,
                               0x100u,
                               0,
                               0,
                               0,
                               0);
    sub_11DAF00(v12, a2);
    return *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  }
  return v13;
}
