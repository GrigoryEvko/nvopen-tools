// Function: sub_EB00F0
// Address: 0xeb00f0
//
__int64 __fastcall sub_EB00F0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r13
  _BOOL8 v4; // rsi
  __int64 result; // rax
  __int64 v6; // rdi
  bool v7; // al
  __int64 v8; // r13
  unsigned int v9; // r11d
  char v10; // al
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // cc
  _QWORD *v14; // rax
  const char *v15; // rax
  unsigned int v16; // [rsp+8h] [rbp-D8h]
  unsigned int v17; // [rsp+8h] [rbp-D8h]
  unsigned int v18; // [rsp+14h] [rbp-CCh] BYREF
  __int64 v19; // [rsp+18h] [rbp-C8h] BYREF
  unsigned int v20[8]; // [rsp+20h] [rbp-C0h] BYREF
  char v21; // [rsp+40h] [rbp-A0h]
  char v22; // [rsp+41h] [rbp-9Fh]
  _QWORD v23[4]; // [rsp+50h] [rbp-90h] BYREF
  char v24; // [rsp+70h] [rbp-70h]
  char v25; // [rsp+71h] [rbp-6Fh]
  _QWORD v26[4]; // [rsp+80h] [rbp-60h] BYREF
  char v27; // [rsp+A0h] [rbp-40h]
  char v28; // [rsp+A1h] [rbp-3Fh]

  v19 = 0;
  v2 = sub_ECD7B0(a1);
  v3 = sub_ECD6A0(v2);
  v22 = 1;
  *(_QWORD *)v20 = "expected integer";
  v21 = 3;
  if ( (unsigned __int8)sub_ECE130(a1, &v19, v20) )
    return 1;
  v4 = 0;
  v25 = 1;
  v23[0] = "file number less than one in '.loc' directive";
  v24 = 3;
  if ( v19 <= 0 )
    v4 = *(_WORD *)(*(_QWORD *)(a1 + 224) + 1904LL) <= 4u;
  if ( (unsigned __int8)sub_ECE070(a1, v4, v3, v23) )
    return 1;
  v28 = 1;
  v6 = *(_QWORD *)(a1 + 224);
  v27 = 3;
  v26[0] = "unassigned file number in '.loc' directive";
  v7 = sub_E70000(v6, v19, 0);
  if ( (unsigned __int8)sub_ECE070(a1, !v7, v3, v26) )
    return 1;
  LODWORD(v8) = 0;
  v9 = 0;
  if ( **(_DWORD **)(a1 + 48) == 4 )
  {
    v11 = sub_ECD7B0(a1);
    if ( *(_DWORD *)(v11 + 32) <= 0x40u )
      v8 = *(_QWORD *)(v11 + 24);
    else
      v8 = **(_QWORD **)(v11 + 24);
    if ( v8 < 0 )
    {
      v28 = 1;
      v15 = "line number less than zero in '.loc' directive";
    }
    else
    {
      sub_EABFE0(a1);
      if ( **(_DWORD **)(a1 + 48) != 4 )
      {
        v9 = 0;
        goto LABEL_9;
      }
      v12 = sub_ECD7B0(a1);
      v13 = *(_DWORD *)(v12 + 32) <= 0x40u;
      v14 = *(_QWORD **)(v12 + 24);
      if ( !v13 )
        v14 = (_QWORD *)*v14;
      if ( (__int64)v14 >= 0 )
      {
        v17 = (unsigned int)v14;
        sub_EABFE0(a1);
        v9 = v17;
        goto LABEL_9;
      }
      v28 = 1;
      v15 = "column position less than zero in '.loc' directive";
    }
    v26[0] = v15;
    v27 = 3;
    return sub_ECE0E0(a1, v26, 0, 0);
  }
LABEL_9:
  v16 = v9;
  v10 = *(_BYTE *)(*(_QWORD *)(a1 + 224) + 1786LL);
  v26[0] = a1;
  v20[0] = 0;
  v26[2] = v20;
  v18 = v10 & 1;
  v23[0] = 0;
  v26[1] = &v18;
  v26[3] = v23;
  result = sub_ECE300(a1, sub_EB8830, v26, 0);
  if ( !(_BYTE)result )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 232) + 688LL))(
      *(_QWORD *)(a1 + 232),
      (unsigned int)v19,
      (unsigned int)v8,
      v16,
      v18,
      v20[0],
      v23[0],
      0,
      0,
      0,
      0);
    return 0;
  }
  return result;
}
