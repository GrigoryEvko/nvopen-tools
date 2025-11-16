// Function: sub_11E06A0
// Address: 0x11e06a0
//
__int64 __fastcall sub_11E06A0(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // eax
  char v6; // al
  unsigned __int8 **v7; // rsi
  char v8; // r15
  char v9; // al
  __int64 result; // rax
  __int64 v11; // r13
  _QWORD *v12; // rdi
  __int64 v13; // rax
  _BYTE *v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rax
  _BYTE *v18; // [rsp+8h] [rbp-88h] BYREF
  __int64 v19; // [rsp+10h] [rbp-80h] BYREF
  __int64 v20; // [rsp+18h] [rbp-78h]
  unsigned __int8 **v21; // [rsp+20h] [rbp-70h] BYREF
  __int64 v22; // [rsp+28h] [rbp-68h]
  char *v23; // [rsp+30h] [rbp-60h] BYREF
  char v24; // [rsp+50h] [rbp-40h]
  char v25; // [rsp+51h] [rbp-3Fh]

  v5 = *(_DWORD *)(a2 + 4);
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v6 = sub_98B0F0(*(_QWORD *)(a2 - 32LL * (v5 & 0x7FFFFFF)), &v19, 1u);
  v7 = (unsigned __int8 **)&v21;
  v8 = v6;
  v9 = sub_98B0F0(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), &v21, 1u);
  if ( v8 && !v20 )
    return sub_AD6530(*(_QWORD *)(a2 + 8), (__int64)v7);
  if ( !v9 )
    return 0;
  if ( !v22 )
    return sub_AD6530(*(_QWORD *)(a2 + 8), (__int64)v7);
  if ( v8 )
  {
    v7 = v21;
    v11 = sub_C934D0(&v19, (unsigned __int8 *)v21, v22, 0);
    if ( v11 != -1 )
    {
      v12 = *(_QWORD **)(a3 + 72);
      v25 = 1;
      v23 = "strpbrk";
      v24 = 3;
      v13 = sub_BCB2E0(v12);
      v14 = (_BYTE *)sub_ACD640(v13, v11, 0);
      v15 = *(_QWORD **)(a3 + 72);
      v18 = v14;
      v16 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v17 = sub_BCB2B0(v15);
      return sub_921130((unsigned int **)a3, v17, v16, &v18, 1, (__int64)&v23, 3u);
    }
    return sub_AD6530(*(_QWORD *)(a2 + 8), (__int64)v7);
  }
  if ( v22 != 1 )
    return 0;
  result = sub_11CA130(
             *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
             *(_BYTE *)v21,
             a3,
             *(__int64 **)(a1 + 24));
  if ( !result )
    return 0;
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return result;
}
