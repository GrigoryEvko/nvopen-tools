// Function: sub_3269740
// Address: 0x3269740
//
__int64 __fastcall sub_3269740(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 v7; // rbx
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // rax
  unsigned int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned int v15; // r14d
  unsigned int v16; // r15d
  __int64 v17; // rdi
  unsigned int v18; // r14d
  __int16 v19; // ax
  __int64 v20; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  int v25; // [rsp+10h] [rbp-90h]
  int v26; // [rsp+14h] [rbp-8Ch]
  __int64 v27; // [rsp+18h] [rbp-88h]
  __int64 v28; // [rsp+20h] [rbp-80h]
  __int64 v29; // [rsp+28h] [rbp-78h]
  __int64 v30; // [rsp+30h] [rbp-70h] BYREF
  __int64 v31; // [rsp+38h] [rbp-68h]
  __int64 v32; // [rsp+40h] [rbp-60h] BYREF
  int v33; // [rsp+48h] [rbp-58h]
  __int64 v34; // [rsp+50h] [rbp-50h] BYREF
  __int64 v35; // [rsp+58h] [rbp-48h]
  __int64 v36; // [rsp+60h] [rbp-40h]
  __int64 v37; // [rsp+68h] [rbp-38h]

  v3 = *(__int64 **)(a1 + 40);
  v4 = *(_QWORD *)(a1 + 80);
  v5 = *v3;
  v6 = v3[6];
  v29 = v3[1];
  v28 = *v3;
  v25 = *((_DWORD *)v3 + 2);
  v7 = v3[5];
  v26 = *((_DWORD *)v3 + 12);
  v8 = *(__int16 **)(a1 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v32 = v4;
  LOWORD(v30) = v9;
  v31 = v10;
  if ( v4 )
    sub_B96E90((__int64)&v32, v4, 1);
  v11 = *(_DWORD *)(a1 + 24);
  v33 = *(_DWORD *)(a1 + 72);
  v12 = sub_33DFBC0(v7, v6, 0, 0);
  v37 = v6;
  v34 = v5;
  v27 = v12;
  v36 = v7;
  v35 = v29;
  if ( (unsigned __int8)sub_33CF1D0(a2, v11, &v34, 2) )
  {
    v34 = 0;
    LODWORD(v35) = 0;
    v5 = sub_33F17F0(a2, 51, &v34, v30, v31);
    if ( v34 )
      sub_B91220((__int64)&v34, v34);
    goto LABEL_21;
  }
  if ( *(_DWORD *)(v28 + 24) == 51 )
  {
LABEL_19:
    v20 = sub_3400BD0(a2, 0, (unsigned int)&v32, v30, v31, 0, 0);
LABEL_20:
    v5 = v20;
    goto LABEL_21;
  }
  v13 = sub_33DFBC0(v5, v29, 0, 0);
  if ( v13 )
  {
    v14 = *(_QWORD *)(v13 + 96);
    v15 = *(_DWORD *)(v14 + 32);
    if ( v15 <= 0x40 )
    {
      if ( !*(_QWORD *)(v14 + 24) )
        goto LABEL_21;
    }
    else if ( v15 == (unsigned int)sub_C444A0(v14 + 24) )
    {
      goto LABEL_21;
    }
  }
  v16 = v11 - 59;
  if ( v26 == v25 && v7 == v28 )
  {
    v20 = sub_3400BD0(a2, v16 <= 1, (unsigned int)&v32, v30, v31, 0, 0);
    goto LABEL_20;
  }
  if ( !v27 )
    goto LABEL_14;
  v17 = *(_QWORD *)(v27 + 96);
  v18 = *(_DWORD *)(v17 + 32);
  if ( v18 > 0x40 )
  {
    if ( (unsigned int)sub_C444A0(v17 + 24) == v18 - 1 )
      goto LABEL_18;
LABEL_14:
    v19 = v30;
    if ( (_WORD)v30 )
    {
      if ( (unsigned __int16)(v30 - 17) <= 0xD3u )
        v19 = word_4456580[(unsigned __int16)v30 - 1];
    }
    else
    {
      if ( !sub_30070B0((__int64)&v30) )
        goto LABEL_31;
      v19 = sub_3009970((__int64)&v30, v29, v22, v23, v24);
    }
    if ( v19 == 2 )
      goto LABEL_18;
LABEL_31:
    v5 = 0;
    goto LABEL_21;
  }
  if ( *(_QWORD *)(v17 + 24) != 1 )
    goto LABEL_14;
LABEL_18:
  if ( v16 > 1 )
    goto LABEL_19;
LABEL_21:
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
  return v5;
}
