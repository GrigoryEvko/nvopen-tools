// Function: sub_326BC50
// Address: 0x326bc50
//
__int64 __fastcall sub_326BC50(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // r14d
  __int64 v5; // r13
  int v6; // edx
  __int64 (*v7)(void); // rax
  unsigned __int16 *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r15
  int v11; // ecx
  __int64 v12; // r8
  int v13; // esi
  __int64 v14; // r12
  __int64 result; // rax
  char v16; // al
  __int64 v17; // r15
  __int64 v18; // rax
  int v19; // ecx
  __int64 v20; // rsi
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r9
  __int64 v25; // r8
  int v26; // r13d
  int v27; // [rsp+4h] [rbp-6Ch]
  int v28; // [rsp+8h] [rbp-68h]
  int v29; // [rsp+8h] [rbp-68h]
  int v30; // [rsp+10h] [rbp-60h]
  int v31; // [rsp+10h] [rbp-60h]
  unsigned int v32; // [rsp+10h] [rbp-60h]
  int v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+10h] [rbp-60h]
  __int64 v35; // [rsp+20h] [rbp-50h] BYREF
  int v36; // [rsp+28h] [rbp-48h]
  __int64 v37; // [rsp+30h] [rbp-40h] BYREF
  int v38; // [rsp+38h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_DWORD *)(a2 + 24);
  v5 = *(_QWORD *)v3;
  v6 = *(_DWORD *)(v3 + 8);
  v7 = *(__int64 (**)(void))(*(_QWORD *)a1[1] + 1744LL);
  if ( v7 != sub_2FE3610 )
  {
    v31 = v6;
    v16 = v7();
    v6 = v31;
    if ( v16 )
      goto LABEL_3;
  }
  if ( *(_DWORD *)(v5 + 24) != 186 )
    goto LABEL_3;
  v17 = *(_QWORD *)(v5 + 40);
  v18 = *(_QWORD *)(v17 + 40);
  v19 = *(_DWORD *)(v18 + 24);
  if ( v19 != 35 && v19 != 11 )
    goto LABEL_3;
  if ( (*(_BYTE *)(v18 + 32) & 8) != 0 )
    goto LABEL_3;
  v20 = *(_QWORD *)(v18 + 96);
  v32 = *(_DWORD *)(v20 + 32);
  if ( v32 <= 0x40 )
  {
    v22 = *(_QWORD *)(v20 + 24);
  }
  else
  {
    v27 = v6;
    v21 = sub_C444A0(v20 + 24);
    v6 = v27;
    if ( v32 - v21 > 0x40 )
      goto LABEL_3;
    v22 = **(_QWORD **)(v20 + 24);
  }
  if ( v22 == 0xFFFF )
  {
    v23 = *(_QWORD *)(a2 + 80);
    v24 = *a1;
    v25 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
    v26 = **(unsigned __int16 **)(a2 + 48);
    v37 = v23;
    if ( v23 )
    {
      v29 = v25;
      v33 = v24;
      sub_B96E90((__int64)&v37, v23, 1);
      LODWORD(v25) = v29;
      LODWORD(v24) = v33;
    }
    v38 = *(_DWORD *)(a2 + 72);
    result = sub_33FAF80(v24, v4, (unsigned int)&v37, v26, v25, v24, *(_OWORD *)v17);
    if ( v37 )
    {
      v34 = result;
      sub_B91220((__int64)&v37, v37);
      return v34;
    }
    return result;
  }
LABEL_3:
  v8 = *(unsigned __int16 **)(a2 + 48);
  v9 = *(_QWORD *)(a2 + 80);
  v37 = v5;
  v38 = v6;
  v10 = *a1;
  v11 = *v8;
  v12 = *((_QWORD *)v8 + 1);
  v35 = v9;
  if ( v9 )
  {
    v28 = v11;
    v30 = v12;
    sub_B96E90((__int64)&v35, v9, 1);
    v11 = v28;
    LODWORD(v12) = v30;
  }
  v13 = *(_DWORD *)(a2 + 24);
  v36 = *(_DWORD *)(a2 + 72);
  v14 = sub_3402EA0(v10, v13, (unsigned int)&v35, v11, v12, 0, (__int64)&v37, 1);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  return v14;
}
