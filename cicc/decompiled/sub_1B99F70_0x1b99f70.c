// Function: sub_1B99F70
// Address: 0x1b99f70
//
unsigned __int64 __fastcall sub_1B99F70(__int64 *a1, unsigned __int64 *a2, unsigned int *a3)
{
  __int64 *v5; // rdi
  __int64 v7; // r8
  __int64 v8; // r15
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // rsi
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  unsigned __int64 *v13; // rax
  __int64 v14; // r10
  unsigned __int64 *v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // r15
  __int64 v18; // r11
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r15
  _QWORD *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r12
  unsigned __int64 result; // rax
  __int64 v28; // [rsp+0h] [rbp-90h]
  __int64 v29; // [rsp+8h] [rbp-88h]
  __int64 v30; // [rsp+8h] [rbp-88h]
  __int64 v31; // [rsp+10h] [rbp-80h]
  __int64 v32; // [rsp+10h] [rbp-80h]
  __int64 *v33; // [rsp+10h] [rbp-80h]
  __int64 v34; // [rsp+18h] [rbp-78h]
  _QWORD *v35; // [rsp+18h] [rbp-78h]
  __int64 v36[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v37; // [rsp+30h] [rbp-60h]
  unsigned __int64 *v38[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v39; // [rsp+50h] [rbp-40h]

  v5 = a1 + 43;
  v7 = a3[1];
  v8 = *a3;
  v36[0] = (__int64)a2;
  v9 = (unsigned __int64 *)a1[44];
  v10 = (unsigned __int64 *)v5;
  if ( !v9 )
    goto LABEL_8;
  do
  {
    while ( 1 )
    {
      v11 = v9[2];
      v12 = v9[3];
      if ( v9[4] >= (unsigned __int64)a2 )
        break;
      v9 = (unsigned __int64 *)v9[3];
      if ( !v12 )
        goto LABEL_6;
    }
    v10 = v9;
    v9 = (unsigned __int64 *)v9[2];
  }
  while ( v11 );
LABEL_6:
  if ( v5 != (__int64 *)v10 && v10[4] <= (unsigned __int64)a2 )
  {
    v31 = v8;
  }
  else
  {
LABEL_8:
    v34 = v7;
    v38[0] = (unsigned __int64 *)v36;
    v13 = sub_1B99EB0(a1 + 42, v10, v38);
    v7 = v34;
    v10 = v13;
    v31 = *a3;
  }
  v14 = *(_QWORD *)(*(_QWORD *)(v10[5] + 48 * v8) + 8 * v7);
  v38[0] = a2;
  v35 = a1 + 36;
  v29 = v14;
  v15 = sub_1B99AC0(a1 + 36, (unsigned __int64 *)v38);
  v16 = (_QWORD *)a1[15];
  v17 = a3[1];
  v18 = *(_QWORD *)(*v15 + 8 * v31);
  v37 = 257;
  v32 = v18;
  v19 = sub_1643350(v16);
  v20 = sub_159C470(v19, v17, 0);
  if ( *(_BYTE *)(v32 + 16) > 0x10u || *(_BYTE *)(v29 + 16) > 0x10u || *(_BYTE *)(v20 + 16) > 0x10u )
  {
    v28 = v29;
    v30 = v20;
    v39 = 257;
    v22 = sub_1648A60(56, 3u);
    v21 = (__int64)v22;
    if ( v22 )
      sub_15FA480((__int64)v22, (__int64 *)v32, v28, v30, (__int64)v38, 0);
    v23 = a1[13];
    if ( v23 )
    {
      v33 = (__int64 *)a1[14];
      sub_157E9D0(v23 + 40, v21);
      v24 = *v33;
      v25 = *(_QWORD *)(v21 + 24) & 7LL;
      *(_QWORD *)(v21 + 32) = v33;
      v24 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v21 + 24) = v24 | v25;
      *(_QWORD *)(v24 + 8) = v21 + 24;
      *v33 = *v33 & 7 | (v21 + 24);
    }
    sub_164B780(v21, v36);
    sub_12A86E0(a1 + 12, v21);
  }
  else
  {
    v21 = sub_15A3890((__int64 *)v32, v29, v20, 0);
  }
  v26 = *a3;
  v38[0] = a2;
  result = *sub_1B99AC0(v35, (unsigned __int64 *)v38);
  *(_QWORD *)(result + 8 * v26) = v21;
  return result;
}
