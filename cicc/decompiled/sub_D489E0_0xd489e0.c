// Function: sub_D489E0
// Address: 0xd489e0
//
__int64 __fastcall sub_D489E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // rdx
  bool v11; // zf
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-B0h]
  __int64 v16; // [rsp+8h] [rbp-A8h]
  __int64 v17; // [rsp+10h] [rbp-A0h]
  __int64 v18; // [rsp+10h] [rbp-A0h]
  __int64 v19; // [rsp+18h] [rbp-98h]
  __int64 v20; // [rsp+20h] [rbp-90h]
  __int64 v21; // [rsp+20h] [rbp-90h]
  _QWORD v23[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v24; // [rsp+40h] [rbp-70h]
  int v25; // [rsp+48h] [rbp-68h]
  __int64 v26; // [rsp+50h] [rbp-60h]
  __int64 v27; // [rsp+58h] [rbp-58h]
  _BYTE *v28; // [rsp+60h] [rbp-50h]
  __int64 v29; // [rsp+68h] [rbp-48h]
  _BYTE v30[64]; // [rsp+70h] [rbp-40h] BYREF

  v4 = a2;
  v23[0] = 6;
  v23[1] = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = v30;
  v29 = 0x200000000LL;
  if ( !(unsigned __int8)sub_10238A0(a3, a2, a4, v23, 0, 0) || !v24 || !v27 )
    goto LABEL_4;
  v20 = v26;
  if ( (*(_BYTE *)(v27 + 7) & 0x40) != 0 )
    v7 = *(__int64 **)(v27 - 8);
  else
    v7 = (__int64 *)(v27 - 32LL * (*(_DWORD *)(v27 + 4) & 0x7FFFFFF));
  a2 = v7[4];
  v15 = v27;
  v16 = v24;
  v19 = a2;
  v17 = *v7;
  v8 = sub_DD8400(a4, a2);
  v9 = v16;
  v10 = v15;
  if ( v20 != v8 )
  {
    a2 = v17;
    v11 = v20 == sub_DD8400(a4, v17);
    v12 = 0;
    if ( v11 )
      v12 = v17;
    v10 = v15;
    v9 = v16;
    v19 = v12;
  }
  v18 = v10;
  v21 = v9;
  v13 = sub_D48970(v4);
  if ( !v13 )
    goto LABEL_4;
  a2 = *(_QWORD *)(v13 - 64);
  v14 = *(_QWORD *)(v13 - 32);
  if ( !a2 || !v14 )
    goto LABEL_4;
  if ( a2 == a3 || v18 == a2 )
  {
LABEL_27:
    a2 = v14;
    goto LABEL_24;
  }
  if ( a3 == v14 )
  {
    v14 = a2;
    goto LABEL_27;
  }
  if ( v18 != v14 )
  {
LABEL_4:
    *(_BYTE *)(a1 + 48) = 0;
    goto LABEL_5;
  }
LABEL_24:
  *(_QWORD *)a1 = v4;
  *(_QWORD *)(a1 + 8) = v21;
  *(_QWORD *)(a1 + 16) = v18;
  *(_QWORD *)(a1 + 24) = v19;
  *(_QWORD *)(a1 + 32) = a2;
  *(_QWORD *)(a1 + 40) = a4;
  *(_BYTE *)(a1 + 48) = 1;
LABEL_5:
  if ( v28 != v30 )
    _libc_free(v28, a2);
  if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
    sub_BD60C0(v23);
  return a1;
}
