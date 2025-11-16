// Function: sub_12A2620
// Address: 0x12a2620
//
_QWORD *__fastcall sub_12A2620(__int64 a1, __int64 a2, __int64 a3)
{
  const char *v4; // r12
  __int64 v5; // r13
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
  int v17; // [rsp+14h] [rbp-7Ch]
  __int64 v18; // [rsp+20h] [rbp-70h]
  _BOOL4 v19; // [rsp+28h] [rbp-68h]
  _BOOL4 v20; // [rsp+2Ch] [rbp-64h]
  int v21; // [rsp+34h] [rbp-5Ch] BYREF
  __int64 v22; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v23; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v24[8]; // [rsp+50h] [rbp-40h] BYREF

  sub_129EFB0(&v23, a1, a3);
  v4 = (const char *)sub_129E180((__int64)v23, a3);
  sub_129E300(*(_DWORD *)(a1 + 480), (char *)&v21);
  v5 = sub_129F850(a1, *(_DWORD *)(a1 + 480));
  v20 = 1;
  v18 = sub_12A0C10(a1, *(_QWORD *)(a3 + 152));
  v19 = unk_4D04660 != 0;
  v17 = v21;
  if ( (*(_BYTE *)(a3 + 202) & 1) == 0 )
    v20 = (*(_BYTE *)(a2 + 32) & 0xF) == 7;
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
  v12 = sub_15A7010((int)a1 + 16, v5, (_DWORD)v4, v8, v9, v10, v5, v17, v18, v20, 1, v17, 0, v19, 0, 0, 0);
  v13 = *(_DWORD *)(a1 + 480);
  v22 = v12;
  if ( v13 && *(_WORD *)(a1 + 484) )
    sub_1627150(a2, v12);
  sub_129F3B0((__int64 *)(a1 + 496), &v22);
  result = v24;
  if ( v23 != v24 )
    return (_QWORD *)j_j___libc_free_0(v23, v24[0] + 1LL);
  return result;
}
