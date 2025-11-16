// Function: sub_2055E10
// Address: 0x2055e10
//
unsigned __int64 __fastcall sub_2055E10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8,
        char a9)
{
  char v13; // al
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int64 result; // rax
  __int64 v20; // r12
  __int64 v21; // rsi
  __int64 v22; // rdi
  unsigned int v23; // r8d
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // [rsp+0h] [rbp-A0h]
  int v31; // [rsp+0h] [rbp-A0h]
  __int64 v33; // [rsp+10h] [rbp-90h] BYREF
  int v34; // [rsp+18h] [rbp-88h]
  int v35; // [rsp+20h] [rbp-80h] BYREF
  __int64 v36; // [rsp+28h] [rbp-78h]
  __int64 v37; // [rsp+30h] [rbp-70h]
  __int64 v38; // [rsp+38h] [rbp-68h]
  __int64 v39; // [rsp+40h] [rbp-60h]
  __int64 v40; // [rsp+48h] [rbp-58h]
  __int64 v41; // [rsp+50h] [rbp-50h]
  __int64 v42; // [rsp+58h] [rbp-48h] BYREF
  int v43; // [rsp+60h] [rbp-40h]
  int v44; // [rsp+68h] [rbp-38h]
  unsigned int v45; // [rsp+6Ch] [rbp-34h]

  v13 = *(_BYTE *)(a2 + 16);
  if ( (unsigned __int8)(v13 - 75) > 1u )
    goto LABEL_4;
  if ( a5 == a6 )
    goto LABEL_22;
  v30 = *(_QWORD *)(a5 + 40);
  if ( !(unsigned __int8)sub_2052D80(a1, *(_QWORD *)(a2 - 48), v30)
    || !(unsigned __int8)sub_2052D80(a1, *(_QWORD *)(a2 - 24), v30) )
  {
LABEL_4:
    v14 = *(_QWORD *)a1;
    v15 = *(_DWORD *)(a1 + 536);
    v33 = 0;
    v34 = v15;
    if ( v14 )
    {
      if ( &v33 != (__int64 *)(v14 + 48) )
      {
        v16 = *(_QWORD *)(v14 + 48);
        v33 = v16;
        if ( v16 )
          sub_1623A60((__int64)&v33, v16, 2);
      }
    }
    v17 = sub_159C4F0(*(__int64 **)(*(_QWORD *)(a1 + 552) + 48LL));
    v35 = a9 == 0 ? 17 : 22;
    v38 = v17;
    v36 = a2;
    v37 = 0;
    v39 = a3;
    v40 = a4;
    v41 = a5;
    v42 = v33;
    if ( v33 )
    {
      sub_1623A60((__int64)&v42, v33, 2);
      v18 = v33;
      v43 = v34;
      v44 = a7;
      result = a8;
      v45 = a8;
      if ( !v33 )
        goto LABEL_11;
      goto LABEL_10;
    }
    goto LABEL_30;
  }
  v13 = *(_BYTE *)(a2 + 16);
LABEL_22:
  v22 = *(_WORD *)(a2 + 18) & 0x7FFF;
  if ( v13 == 75 )
  {
    if ( a9 )
      v22 = (unsigned int)sub_15FF0F0(v22);
    v23 = sub_20C8390(v22);
  }
  else
  {
    if ( a9 )
      v22 = (unsigned int)sub_15FF0F0(v22);
    v23 = sub_20C82F0(v22);
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 544) + 792LL) & 8) != 0 )
      v23 = sub_20C8300(v23);
  }
  v24 = *(_QWORD *)a1;
  v34 = *(_DWORD *)(a1 + 536);
  if ( !v24 || &v33 == (__int64 *)(v24 + 48) || (v27 = *(_QWORD *)(v24 + 48), (v33 = v27) == 0) )
  {
    v25 = *(_QWORD *)(a2 - 24);
    v26 = *(_QWORD *)(a2 - 48);
    v35 = v23;
    v37 = 0;
    v38 = v25;
    v36 = v26;
    v39 = a3;
    v40 = a4;
    v41 = a5;
    v42 = 0;
    goto LABEL_30;
  }
  v31 = v23;
  sub_1623A60((__int64)&v33, v27, 2);
  v28 = *(_QWORD *)(a2 - 24);
  v39 = a3;
  v29 = *(_QWORD *)(a2 - 48);
  v41 = a5;
  v38 = v28;
  v36 = v29;
  v35 = v31;
  v37 = 0;
  v40 = a4;
  v42 = v33;
  if ( !v33 )
  {
LABEL_30:
    v20 = *(_QWORD *)(a1 + 592);
    v43 = v34;
    v44 = a7;
    result = a8;
    v45 = a8;
    if ( v20 != *(_QWORD *)(a1 + 600) )
      goto LABEL_12;
LABEL_31:
    result = sub_2055B00((__int64 *)(a1 + 584), v20, (__int64)&v35);
    goto LABEL_17;
  }
  sub_1623A60((__int64)&v42, v33, 2);
  v18 = v33;
  v43 = v34;
  v44 = a7;
  result = a8;
  v45 = a8;
  if ( v33 )
LABEL_10:
    result = sub_161E7C0((__int64)&v33, v18);
LABEL_11:
  v20 = *(_QWORD *)(a1 + 592);
  if ( v20 == *(_QWORD *)(a1 + 600) )
    goto LABEL_31;
LABEL_12:
  if ( v20 )
  {
    *(_DWORD *)v20 = v35;
    *(_QWORD *)(v20 + 8) = v36;
    *(_QWORD *)(v20 + 16) = v37;
    *(_QWORD *)(v20 + 24) = v38;
    *(_QWORD *)(v20 + 32) = v39;
    *(_QWORD *)(v20 + 40) = v40;
    *(_QWORD *)(v20 + 48) = v41;
    v21 = v42;
    *(_QWORD *)(v20 + 56) = v42;
    if ( v21 )
      sub_1623A60(v20 + 56, v21, 2);
    *(_DWORD *)(v20 + 64) = v43;
    *(_DWORD *)(v20 + 72) = v44;
    result = v45;
    *(_DWORD *)(v20 + 76) = v45;
    v20 = *(_QWORD *)(a1 + 592);
  }
  *(_QWORD *)(a1 + 592) = v20 + 80;
LABEL_17:
  if ( v42 )
    return sub_161E7C0((__int64)&v42, v42);
  return result;
}
