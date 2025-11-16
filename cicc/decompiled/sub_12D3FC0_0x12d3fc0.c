// Function: sub_12D3FC0
// Address: 0x12d3fc0
//
__int64 __fastcall sub_12D3FC0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rcx
  _QWORD *v3; // rdx
  _QWORD *v4; // rax
  _BYTE *v5; // r14
  _BYTE *v6; // r13
  _QWORD *v7; // rdi
  __int64 (__fastcall *v8)(__int64); // rax
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rdi
  _BYTE *v12; // r15
  _BYTE *v13; // r14
  _QWORD *v14; // rdi
  __int64 (__fastcall *v15)(__int64); // rax
  __int64 v16; // rdi
  _QWORD *v18; // [rsp+0h] [rbp-E0h]
  _QWORD *v19; // [rsp+8h] [rbp-D8h]
  _QWORD *v20; // [rsp+10h] [rbp-D0h]
  unsigned int v21; // [rsp+24h] [rbp-BCh]
  _QWORD *v22; // [rsp+28h] [rbp-B8h]
  _QWORD *v23; // [rsp+30h] [rbp-B0h] BYREF
  _QWORD *v24; // [rsp+38h] [rbp-A8h]
  _QWORD *v25; // [rsp+40h] [rbp-A0h]
  _QWORD *v26; // [rsp+48h] [rbp-98h]
  _QWORD *v27; // [rsp+50h] [rbp-90h]
  _QWORD *v28; // [rsp+58h] [rbp-88h]
  _QWORD *v29; // [rsp+60h] [rbp-80h]
  _QWORD *v30; // [rsp+68h] [rbp-78h]
  _BYTE v31[16]; // [rsp+70h] [rbp-70h] BYREF
  __int64 (__fastcall *v32)(__int64); // [rsp+80h] [rbp-60h]
  __int64 v33; // [rsp+88h] [rbp-58h]
  __int64 (__fastcall *v34)(__int64); // [rsp+90h] [rbp-50h]
  __int64 v35; // [rsp+98h] [rbp-48h]
  __int64 (__fastcall *v36)(__int64 *); // [rsp+A0h] [rbp-40h]
  __int64 v37; // [rsp+A8h] [rbp-38h]

  v2 = (_QWORD *)a1[6];
  v3 = (_QWORD *)a1[2];
  v4 = (_QWORD *)a1[4];
  v19 = a1 + 7;
  v18 = a1 + 5;
  v20 = a1 + 1;
  v22 = a1 + 3;
  v23 = (_QWORD *)a1[8];
  v24 = a1 + 7;
  v25 = v2;
  v26 = a1 + 5;
  v27 = v3;
  v28 = a1 + 1;
  v29 = v4;
  v30 = a1 + 3;
  v21 = 0;
  if ( a1 + 3 == v4 )
    goto LABEL_15;
  while ( 1 )
  {
    v5 = v31;
    v33 = 0;
    v6 = v31;
    v7 = &v23;
    v32 = sub_12D3C60;
    v35 = 0;
    v34 = sub_12D3C80;
    v37 = 0;
    v36 = sub_12D3CA0;
    v8 = sub_12D3C40;
    if ( ((unsigned __int8)sub_12D3C40 & 1) == 0 )
      goto LABEL_4;
    while ( 1 )
    {
      v8 = *(__int64 (__fastcall **)(__int64))((char *)v8 + *v7 - 1);
LABEL_4:
      v9 = v8((__int64)v7);
      v10 = v9;
      if ( v9 )
        break;
      while ( 1 )
      {
        v11 = *((_QWORD *)v6 + 3);
        v8 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v6 + 2);
        v5 += 16;
        v6 = v5;
        v7 = (_QWORD **)((char *)&v23 + v11);
        if ( ((unsigned __int8)v8 & 1) != 0 )
          break;
        v9 = v8((__int64)v7);
        v10 = v9;
        if ( v9 )
          goto LABEL_7;
      }
    }
LABEL_7:
    if ( !*(_BYTE *)(v9 + 16) && !(unsigned __int8)sub_15E4F60(v9) )
    {
      v21 += sub_12D3D20(v10);
      if ( v21 > *(_DWORD *)(a2 + 3984) )
        return 1;
    }
    v12 = v31;
    v33 = 0;
    v35 = 0;
    v13 = v31;
    v14 = &v23;
    v32 = sub_12D3BB0;
    v37 = 0;
    v34 = sub_12D3BE0;
    v36 = sub_12D3C10;
    v15 = sub_12D3B80;
    if ( ((unsigned __int8)sub_12D3B80 & 1) == 0 )
      goto LABEL_11;
    while ( 1 )
    {
      v15 = *(__int64 (__fastcall **)(__int64))((char *)v15 + *v14 - 1);
LABEL_11:
      if ( (unsigned __int8)v15((__int64)v14) )
        break;
      while ( 1 )
      {
        v16 = *((_QWORD *)v13 + 3);
        v15 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v13 + 2);
        v12 += 16;
        v13 = v12;
        v14 = (_QWORD **)((char *)&v23 + v16);
        if ( ((unsigned __int8)v15 & 1) != 0 )
          break;
        if ( (unsigned __int8)v15((__int64)v14) )
          goto LABEL_14;
      }
    }
LABEL_14:
    if ( v22 == v29 )
    {
LABEL_15:
      if ( v22 == v30 && v20 == v27 && v20 == v28 && v18 == v25 && v18 == v26 && v19 == v23 && v19 == v24 )
        return 0;
    }
  }
}
