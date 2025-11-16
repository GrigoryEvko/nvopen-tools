// Function: sub_3889300
// Address: 0x3889300
//
__int64 __fastcall sub_3889300(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool v4; // zf
  unsigned __int64 v5; // rsi
  unsigned int v7; // r12d
  unsigned __int64 v8; // r13
  int v9; // eax
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-98h]
  _QWORD v13[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v14[2]; // [rsp+20h] [rbp-80h] BYREF
  __int16 v15; // [rsp+30h] [rbp-70h]
  _QWORD v16[2]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v17; // [rsp+50h] [rbp-50h]
  _QWORD v18[2]; // [rsp+60h] [rbp-40h] BYREF
  __int16 v19; // [rsp+70h] [rbp-30h]

  v4 = *(_DWORD *)(a1 + 64) == 390;
  v13[0] = a2;
  v13[1] = a3;
  if ( !v4 || !*(_BYTE *)(a1 + 164) )
  {
    v5 = *(_QWORD *)(a1 + 56);
    v19 = 259;
    v18[0] = "expected unsigned integer";
    return sub_38814C0(a1 + 8, v5, (__int64)v18);
  }
  v7 = *(_DWORD *)(a1 + 160);
  v8 = *(_QWORD *)(a4 + 16);
  if ( v7 <= 0x40 )
  {
    v11 = *(_QWORD *)(a1 + 152);
    if ( v8 < v11 )
    {
LABEL_6:
      v10 = *(_QWORD *)(a1 + 56);
      v15 = 1283;
      v14[0] = "value for '";
      v14[1] = v13;
      v16[0] = v14;
      v16[1] = "' too large, limit is ";
      v17 = 770;
      v18[1] = a4 + 16;
      v18[0] = v16;
      v19 = 2818;
      return sub_38814C0(a1 + 8, v10, (__int64)v18);
    }
  }
  else
  {
    v12 = a4;
    v9 = sub_16A57B0(a1 + 152);
    a4 = v12;
    if ( v7 - v9 > 0x40 )
      goto LABEL_6;
    v11 = **(_QWORD **)(a1 + 152);
    if ( v8 < v11 )
      goto LABEL_6;
  }
  *(_BYTE *)(a4 + 8) = 1;
  *(_QWORD *)a4 = v11;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  return 0;
}
