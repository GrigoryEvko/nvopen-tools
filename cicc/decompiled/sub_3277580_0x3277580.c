// Function: sub_3277580
// Address: 0x3277580
//
__int64 __fastcall sub_3277580(__int64 a1, __int64 a2, int a3, __int64 a4, int a5, int a6)
{
  int v6; // r15d
  __int64 result; // rax
  __int64 v8; // rax
  int v11; // edi
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // r9d
  __int64 v17; // r11
  unsigned int v18; // edx
  unsigned __int64 v19; // r13
  __int64 v20; // rsi
  int v21; // ecx
  __int64 v22; // r8
  unsigned int v23; // edx
  __int128 v24; // [rsp-20h] [rbp-D0h]
  unsigned int v25; // [rsp+4h] [rbp-ACh]
  __int64 v26; // [rsp+8h] [rbp-A8h]
  __int64 v27; // [rsp+10h] [rbp-A0h]
  __int64 v28; // [rsp+18h] [rbp-98h]
  int v29; // [rsp+20h] [rbp-90h]
  unsigned int v30; // [rsp+28h] [rbp-88h]
  int v31; // [rsp+28h] [rbp-88h]
  __int64 v33; // [rsp+30h] [rbp-80h]
  int v34; // [rsp+38h] [rbp-78h]
  __int128 v35; // [rsp+40h] [rbp-70h]
  __int64 v36; // [rsp+40h] [rbp-70h]
  __int64 v37; // [rsp+70h] [rbp-40h] BYREF
  int v38; // [rsp+78h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 24);
  if ( v6 != *(_DWORD *)(a2 + 24) )
    return 0;
  if ( *(_DWORD *)(a4 + 24) != v6 )
    return 0;
  v8 = *(_QWORD *)(a2 + 56);
  if ( !v8 )
    return 0;
  v11 = 1;
  do
  {
    if ( a3 == *(_DWORD *)(v8 + 8) )
    {
      if ( !v11 )
        return 0;
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        goto LABEL_14;
      if ( a3 == *(_DWORD *)(v8 + 8) )
        return 0;
      v11 = 0;
    }
    v8 = *(_QWORD *)(v8 + 32);
  }
  while ( v8 );
  if ( v11 == 1 )
    return 0;
LABEL_14:
  v12 = *(_QWORD *)(a4 + 56);
  if ( !v12 )
    return 0;
  v13 = 1;
  do
  {
    if ( a5 == *(_DWORD *)(v12 + 8) )
    {
      if ( !v13 )
        return 0;
      v12 = *(_QWORD *)(v12 + 32);
      if ( !v12 )
        goto LABEL_23;
      if ( *(_DWORD *)(v12 + 8) == a5 )
        return 0;
      v13 = 0;
    }
    v12 = *(_QWORD *)(v12 + 32);
  }
  while ( v12 );
  if ( v13 == 1 )
    return 0;
LABEL_23:
  v14 = *(_QWORD *)(a4 + 40);
  v25 = *(_DWORD *)(v14 + 8);
  v26 = *(_QWORD *)v14;
  v28 = *(_QWORD *)(v14 + 40);
  v27 = *(_QWORD *)(v14 + 48);
  v30 = *(_DWORD *)(v14 + 48);
  v15 = sub_32772A0(a1, a2, a3, *(_QWORD *)v14, *(_QWORD *)(v14 + 8), a6);
  v16 = a6;
  v17 = v15;
  v19 = v18;
  if ( !v15 )
  {
    v17 = sub_32772A0(a1, a2, a3, v28, v27, a6);
    v19 = v23 | v19 & 0xFFFFFFFF00000000LL;
    if ( v17 )
    {
      v16 = a6;
      *(_QWORD *)&v35 = v26;
      *((_QWORD *)&v35 + 1) = v25;
      goto LABEL_25;
    }
    return 0;
  }
  *(_QWORD *)&v35 = v28;
  *((_QWORD *)&v35 + 1) = v30;
LABEL_25:
  v20 = *(_QWORD *)(a1 + 80);
  v21 = **(unsigned __int16 **)(a1 + 48);
  v22 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
  v37 = v20;
  if ( v20 )
  {
    v33 = v17;
    v29 = v21;
    v31 = v16;
    v34 = v22;
    sub_B96E90((__int64)&v37, v20, 1);
    v21 = v29;
    v16 = v31;
    v17 = v33;
    LODWORD(v22) = v34;
  }
  *((_QWORD *)&v24 + 1) = v19;
  *(_QWORD *)&v24 = v17;
  v38 = *(_DWORD *)(a1 + 72);
  result = sub_3406EB0(v16, v6, (unsigned int)&v37, v21, v22, v16, v24, v35);
  if ( v37 )
  {
    v36 = result;
    sub_B91220((__int64)&v37, v37);
    return v36;
  }
  return result;
}
