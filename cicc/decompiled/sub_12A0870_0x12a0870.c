// Function: sub_12A0870
// Address: 0x12a0870
//
_QWORD *__fastcall sub_12A0870(__int64 a1, __int64 a2, __int64 a3)
{
  const char *v4; // r12
  __int64 v5; // rax
  int v6; // eax
  int v7; // edx
  int v8; // ecx
  int v9; // r8d
  int v10; // r9d
  int v11; // eax
  __int64 v12; // rax
  int v13; // edx
  _QWORD *result; // rax
  int v15; // [rsp+0h] [rbp-90h]
  int v16; // [rsp+8h] [rbp-88h]
  __int64 v17; // [rsp+10h] [rbp-80h]
  __int64 v18; // [rsp+18h] [rbp-78h]
  int v19; // [rsp+24h] [rbp-6Ch]
  _BOOL4 v20; // [rsp+28h] [rbp-68h]
  _BOOL4 v21; // [rsp+2Ch] [rbp-64h]
  int v22; // [rsp+34h] [rbp-5Ch] BYREF
  __int64 v23; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v24; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v25[8]; // [rsp+50h] [rbp-40h] BYREF

  sub_129EFB0(&v24, a1, a3);
  v4 = (const char *)sub_129E180((__int64)v24, a3);
  sub_129E300(*(_DWORD *)(a1 + 480), (char *)&v22);
  v18 = sub_129F850(a1, *(_DWORD *)(a1 + 480));
  v5 = sub_15A66D0(a1 + 16, 0, 0);
  v21 = 1;
  v17 = sub_15A5D90(a1 + 16, v5, 0, 0);
  v20 = unk_4D04660 != 0;
  v19 = v22;
  if ( (*(_BYTE *)(a3 + 202) & 1) == 0 )
    v21 = (*(_BYTE *)(a2 + 32) & 0xF) == 7;
  v6 = sub_1649960(a2);
  v8 = 0;
  v9 = v6;
  v10 = v7;
  if ( v4 )
  {
    v15 = v6;
    v16 = v7;
    v11 = strlen(v4);
    v9 = v15;
    v10 = v16;
    v8 = v11;
  }
  v12 = sub_15A7010((int)a1 + 16, v18, (_DWORD)v4, v8, v9, v10, v18, v19, v17, v21, 1, v19, 0, v20, 0, 0, 0);
  v13 = *(_DWORD *)(a1 + 480);
  v23 = v12;
  if ( v13 && *(_WORD *)(a1 + 484) )
    sub_1627150(a2, v12);
  sub_129F3B0((__int64 *)(a1 + 496), &v23);
  result = v25;
  if ( v24 != v25 )
    return (_QWORD *)j_j___libc_free_0(v24, v25[0] + 1LL);
  return result;
}
