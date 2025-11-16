// Function: sub_38EE960
// Address: 0x38ee960
//
__int64 __fastcall sub_38EE960(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // r13
  _BOOL8 v5; // rsi
  __int64 result; // rax
  __int64 v7; // rdi
  bool v8; // al
  __int64 v9; // r13
  unsigned int v10; // r11d
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  bool v15; // cc
  _QWORD *v16; // rax
  const char *v17; // rax
  unsigned int v18; // [rsp+8h] [rbp-A8h]
  unsigned int v19; // [rsp+8h] [rbp-A8h]
  unsigned int v20; // [rsp+14h] [rbp-9Ch] BYREF
  __int64 v21; // [rsp+18h] [rbp-98h] BYREF
  unsigned int v22[4]; // [rsp+20h] [rbp-90h] BYREF
  char v23; // [rsp+30h] [rbp-80h]
  char v24; // [rsp+31h] [rbp-7Fh]
  _QWORD v25[2]; // [rsp+40h] [rbp-70h] BYREF
  char v26; // [rsp+50h] [rbp-60h]
  char v27; // [rsp+51h] [rbp-5Fh]
  _QWORD v28[2]; // [rsp+60h] [rbp-50h] BYREF
  unsigned int *v29; // [rsp+70h] [rbp-40h]
  _QWORD *v30; // [rsp+78h] [rbp-38h]

  v21 = 0;
  v2 = sub_3909460(a1);
  v3 = sub_39092A0(v2);
  v24 = 1;
  v4 = v3;
  v23 = 3;
  *(_QWORD *)v22 = "unexpected token in '.loc' directive";
  if ( (unsigned __int8)sub_3909D40(a1, &v21, v22) )
    return 1;
  v5 = 0;
  v27 = 1;
  v25[0] = "file number less than one in '.loc' directive";
  v26 = 3;
  if ( v21 <= 0 )
    v5 = *(_WORD *)(*(_QWORD *)(a1 + 320) + 1160LL) <= 4u;
  if ( (unsigned __int8)sub_3909C80(a1, v5, v4, v25) )
    return 1;
  v7 = *(_QWORD *)(a1 + 320);
  LOWORD(v29) = 259;
  v28[0] = "unassigned file number in '.loc' directive";
  v8 = sub_38C4070(v7, v21, 0);
  if ( (unsigned __int8)sub_3909C80(a1, !v8, v4, v28) )
    return 1;
  LODWORD(v9) = 0;
  v10 = 0;
  if ( **(_DWORD **)(a1 + 152) == 4 )
  {
    v11 = sub_3909460(a1);
    if ( *(_DWORD *)(v11 + 32) <= 0x40u )
      v9 = *(_QWORD *)(v11 + 24);
    else
      v9 = **(_QWORD **)(v11 + 24);
    if ( v9 < 0 )
    {
      BYTE1(v29) = 1;
      v17 = "line number less than zero in '.loc' directive";
    }
    else
    {
      sub_38EB180(a1);
      if ( **(_DWORD **)(a1 + 152) != 4 )
      {
        v10 = 0;
        goto LABEL_9;
      }
      v14 = sub_3909460(a1);
      v15 = *(_DWORD *)(v14 + 32) <= 0x40u;
      v16 = *(_QWORD **)(v14 + 24);
      if ( !v15 )
        v16 = (_QWORD *)*v16;
      if ( (__int64)v16 >= 0 )
      {
        v19 = (unsigned int)v16;
        sub_38EB180(a1);
        v10 = v19;
        goto LABEL_9;
      }
      BYTE1(v29) = 1;
      v17 = "column position less than zero in '.loc' directive";
    }
    v28[0] = v17;
    LOBYTE(v29) = 3;
    return sub_3909CF0(a1, v28, 0, 0, v12, v13);
  }
LABEL_9:
  v28[0] = a1;
  v18 = v10;
  v20 = 1;
  v22[0] = 0;
  v25[0] = 0;
  v28[1] = &v20;
  v29 = v22;
  v30 = v25;
  result = sub_3909F10(a1, sub_38F35B0, v28, 0);
  if ( !(_BYTE)result )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 328) + 584LL))(
      *(_QWORD *)(a1 + 328),
      (unsigned int)v21,
      (unsigned int)v9,
      v18,
      v20,
      v22[0],
      v25[0],
      0,
      0);
    return 0;
  }
  return result;
}
