// Function: sub_3204E90
// Address: 0x3204e90
//
__int64 __fastcall sub_3204E90(_QWORD *a1, unsigned __int64 *a2)
{
  __int16 v3; // bx
  int v4; // r13d
  __int16 v5; // bx
  __int16 v6; // ax
  unsigned __int8 v7; // al
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned int v11; // r13d
  unsigned __int64 v13; // [rsp+0h] [rbp-A0h]
  __int16 v14; // [rsp+Ch] [rbp-94h]
  unsigned __int64 v15[2]; // [rsp+10h] [rbp-90h] BYREF
  __int64 v16; // [rsp+20h] [rbp-80h] BYREF
  _WORD v17[2]; // [rsp+30h] [rbp-70h] BYREF
  _WORD v18[4]; // [rsp+34h] [rbp-6Ch]
  int v19; // [rsp+3Ch] [rbp-64h]
  unsigned __int64 v20; // [rsp+40h] [rbp-60h]
  unsigned __int64 v21; // [rsp+48h] [rbp-58h]
  __int64 v22; // [rsp+50h] [rbp-50h]
  __int64 v23; // [rsp+58h] [rbp-48h]
  unsigned __int64 v24; // [rsp+60h] [rbp-40h]

  v3 = sub_31F58C0((__int64)a2);
  sub_3204160((__int64)v17, (__int64)a1, (__int64)a2);
  v4 = v19;
  v14 = v18[0];
  v6 = v3;
  v5 = v3 | 0x410;
  HIBYTE(v6) |= 4u;
  if ( !LOBYTE(v17[0]) )
    v5 = v6;
  v13 = a2[3] >> 3;
  sub_3205740(v15, a1, a2);
  v7 = *((_BYTE *)a2 - 16);
  if ( (v7 & 2) != 0 )
  {
    v8 = *(_QWORD *)(*(a2 - 4) + 56);
    if ( v8 )
    {
LABEL_5:
      v8 = sub_B91420(v8);
      goto LABEL_6;
    }
  }
  else
  {
    v8 = a2[-((v7 >> 2) & 0xF) + 5];
    if ( v8 )
      goto LABEL_5;
  }
  v9 = 0;
LABEL_6:
  v22 = v8;
  v17[0] = 5382;
  *(_DWORD *)&v18[1] = v4;
  v17[1] = v14;
  v18[0] = v5;
  v20 = v15[0];
  v23 = v9;
  v21 = v15[1];
  v24 = v13;
  v10 = sub_370A1A0(a1 + 81, v17);
  v11 = sub_3707F80(a1 + 79, v10);
  sub_31FDA50(a1, (__int64)a2, v11);
  sub_31FBCA0(a1, a2);
  if ( (__int64 *)v15[0] != &v16 )
    j_j___libc_free_0(v15[0]);
  return v11;
}
