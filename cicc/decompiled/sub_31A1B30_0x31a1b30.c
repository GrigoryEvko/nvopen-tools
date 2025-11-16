// Function: sub_31A1B30
// Address: 0x31a1b30
//
__int64 __fastcall sub_31A1B30(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // rax
  _BYTE *v5; // r15
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // r14
  unsigned __int16 v8; // ax
  unsigned __int16 v9; // ax
  unsigned __int8 v10; // dl
  __int64 v11; // rdi
  unsigned int v12; // edx
  bool v13; // al
  __int64 v14; // rax
  unsigned int v15; // r8d
  __int64 v16; // rax
  unsigned int v17; // r8d
  char v18; // al
  __int64 result; // rax
  unsigned __int8 v20; // [rsp+16h] [rbp-DAh]
  unsigned __int8 v21; // [rsp+17h] [rbp-D9h]
  char v22; // [rsp+18h] [rbp-D8h]
  unsigned __int8 v23; // [rsp+18h] [rbp-D8h]
  __int64 v24; // [rsp+28h] [rbp-C8h]
  char *v25; // [rsp+30h] [rbp-C0h] BYREF
  char v26; // [rsp+40h] [rbp-B0h] BYREF
  void *v27; // [rsp+B0h] [rbp-40h]

  v3 = (_QWORD *)(a1 + 72);
  v4 = *((_DWORD *)v3 - 17) & 0x7FFFFFF;
  v5 = (_BYTE *)v3[4 * (2 - v4) - 9];
  v6 = v3[4 * (1 - v4) - 9];
  v7 = v3[-4 * v4 - 9];
  v8 = sub_A74840(v3, 1);
  if ( !HIBYTE(v8) )
    LOBYTE(v8) = 0;
  v21 = v8;
  v9 = sub_A74840(v3, 0);
  v10 = 0;
  if ( HIBYTE(v9) )
    v10 = v9;
  v20 = v10;
  v11 = *(_QWORD *)(a1 + 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  v12 = *(_DWORD *)(v11 + 32);
  if ( v12 <= 0x40 )
    v13 = *(_QWORD *)(v11 + 24) == 0;
  else
    v13 = v12 == (unsigned int)sub_C444A0(v11 + 24);
  v22 = !v13;
  sub_23D0AB0((__int64)&v25, a1, 0, 0, 0);
  v14 = *(_QWORD *)(v6 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
    v14 = **(_QWORD **)(v14 + 16);
  v15 = *(_DWORD *)(v14 + 8);
  v16 = *(_QWORD *)(v7 + 8);
  v17 = v15 >> 8;
  if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
    v16 = **(_QWORD **)(v16 + 16);
  if ( *(_DWORD *)(v16 + 8) >> 8 == v17 )
  {
LABEL_12:
    if ( *v5 == 17 )
      sub_319DE50(a1, v6, v7, (__int64)v5, v21, v20, v22, v22, a2);
    else
      sub_319F540(a1, v6, v7, (__int64)v5, v21, v20, v22, v22, a2);
    v18 = 1;
    goto LABEL_15;
  }
  if ( (unsigned __int8)sub_DF9820((__int64)a2) )
  {
    if ( !(unsigned __int8)sub_DF97F0((__int64)a2) )
    {
      v18 = sub_DF97F0((__int64)a2);
      if ( !v18 )
        goto LABEL_15;
    }
    goto LABEL_12;
  }
  BYTE4(v24) = 0;
  if ( *v5 == 17 )
    sub_319B060(a1, v6, v7, (__int64)v5, v21, v20, v22, v22, 0, a2, v24);
  else
    sub_319C1F0(a1, v6, v7, (__int64)v5, v21, v20, v22, v22, 0, a2, v24);
  v18 = 1;
LABEL_15:
  v23 = v18;
  nullsub_61();
  v27 = &unk_49DA100;
  nullsub_63();
  result = v23;
  if ( v25 != &v26 )
  {
    _libc_free((unsigned __int64)v25);
    return v23;
  }
  return result;
}
