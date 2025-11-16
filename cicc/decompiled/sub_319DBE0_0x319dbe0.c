// Function: sub_319DBE0
// Address: 0x319dbe0
//
void __fastcall sub_319DBE0(__int64 a1, _QWORD *a2, __int64 *a3)
{
  char v4; // bl
  __int64 *v6; // r15
  __int64 *v7; // rax
  __int64 v8; // rdx
  _BYTE *v9; // r15
  __int64 v10; // rax
  unsigned int v11; // r13d
  __int64 v12; // rdi
  bool v13; // dl
  unsigned __int16 v14; // ax
  unsigned __int8 v15; // r13
  unsigned __int16 v16; // ax
  unsigned __int8 v17; // r8
  bool v18; // r15
  char v19; // r15
  unsigned int v20; // eax
  unsigned int v21; // r13d
  unsigned int v22; // eax
  unsigned int v23; // r8d
  char v24; // [rsp+4h] [rbp-4Ch]
  __int64 v25; // [rsp+18h] [rbp-38h]

  v4 = 1;
  if ( a3 )
  {
    v6 = sub_DD8400((__int64)a3, *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))));
    v7 = sub_DD8400((__int64)a3, *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
    v4 = sub_DDCB50(a3, 33, v6, v7, a1) ^ 1;
  }
  BYTE4(v25) = 0;
  v8 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v9 = *(_BYTE **)(a1 + 32 * (2 - v8));
  v10 = *(_QWORD *)(a1 + 32 * (3 - v8));
  v11 = *(_DWORD *)(v10 + 32);
  v12 = v10 + 24;
  if ( *v9 == 17 )
  {
    if ( v11 <= 0x40 )
      v13 = *(_QWORD *)(v10 + 24) == 0;
    else
      v13 = (unsigned int)sub_C444A0(v12) == v11;
    v24 = !v13;
    v14 = sub_A74840((_QWORD *)(a1 + 72), 0);
    v15 = v14;
    if ( !HIBYTE(v14) )
      v15 = 0;
    v16 = sub_A74840((_QWORD *)(a1 + 72), 1);
    v17 = 0;
    if ( HIBYTE(v16) )
      v17 = v16;
    sub_319B060(
      a1,
      *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))),
      *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)),
      (__int64)v9,
      v17,
      v15,
      v24,
      v24,
      v4,
      a2,
      v25);
  }
  else
  {
    if ( v11 <= 0x40 )
      v18 = *(_QWORD *)(v10 + 24) == 0;
    else
      v18 = (unsigned int)sub_C444A0(v12) == v11;
    v19 = !v18;
    LOWORD(v20) = sub_A74840((_QWORD *)(a1 + 72), 0);
    v21 = v20;
    if ( !BYTE1(v20) )
      v21 = 0;
    LOWORD(v22) = sub_A74840((_QWORD *)(a1 + 72), 1);
    v23 = 0;
    if ( BYTE1(v22) )
      v23 = v22;
    sub_319C1F0(
      a1,
      *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))),
      *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)),
      *(_QWORD *)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))),
      v23,
      v21,
      v19,
      v19,
      v4,
      a2,
      v25);
  }
}
