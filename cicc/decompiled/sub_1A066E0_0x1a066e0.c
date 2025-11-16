// Function: sub_1A066E0
// Address: 0x1a066e0
//
__int64 __fastcall sub_1A066E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  __int64 v7; // rdx
  char v8; // al
  __int64 v12; // rsi
  __int64 v13; // rax
  _QWORD *v14; // r12
  __int64 v15; // rax
  __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int8 v31; // dl
  __int64 v32; // rsi
  __int64 v33; // rax
  const char *v34; // [rsp+0h] [rbp-60h] BYREF
  __int64 v35; // [rsp+8h] [rbp-58h]
  const char **v36; // [rsp+10h] [rbp-50h] BYREF
  char *v37; // [rsp+18h] [rbp-48h]
  __int16 v38; // [rsp+20h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 16) > 0x10u )
  {
    v12 = 11;
    v13 = sub_19FEFC0(a1, 11, 12);
    v14 = (_QWORD *)v13;
    if ( v13 )
    {
      v15 = sub_1A066E0(*(_QWORD *)(v13 - 48), a2, a3);
      if ( *(v14 - 6) )
      {
        v16 = *(v14 - 5);
        v17 = *(v14 - 4) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v17 = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
      }
      *(v14 - 6) = v15;
      if ( v15 )
      {
        v18 = *(_QWORD *)(v15 + 8);
        *(v14 - 5) = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = (unsigned __int64)(v14 - 5) | *(_QWORD *)(v18 + 16) & 3LL;
        *(v14 - 4) = (v15 + 8) | *(v14 - 4) & 3LL;
        *(_QWORD *)(v15 + 8) = v14 - 6;
      }
      v19 = sub_1A066E0(*(v14 - 3), a2, a3);
      if ( *(v14 - 3) )
      {
        v20 = *(v14 - 2);
        v21 = *(v14 - 1) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v21 = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = *(_QWORD *)(v20 + 16) & 3LL | v21;
      }
      *(v14 - 3) = v19;
      if ( v19 )
      {
        v22 = *(_QWORD *)(v19 + 8);
        *(v14 - 2) = v22;
        if ( v22 )
          *(_QWORD *)(v22 + 16) = (unsigned __int64)(v14 - 2) | *(_QWORD *)(v22 + 16) & 3LL;
        *(v14 - 1) = (v19 + 8) | *(v14 - 1) & 3LL;
        *(_QWORD *)(v19 + 8) = v14 - 3;
      }
      if ( *((_BYTE *)v14 + 16) == 35 )
      {
        sub_15F2310((__int64)v14, 0);
        sub_15F2330((__int64)v14, 0);
      }
      sub_15F22F0(v14, a2);
      v34 = sub_1649960((__int64)v14);
      v35 = v23;
      v36 = &v34;
      v38 = 773;
      v37 = ".neg";
      sub_164B780((__int64)v14, (__int64 *)&v36);
      v36 = (const char **)v14;
      sub_1A062A0(a3, &v36);
      return (__int64)v14;
    }
    v24 = *(_QWORD *)(a1 + 8);
    if ( !v24 )
    {
LABEL_30:
      v34 = sub_1649960(a1);
      v38 = 773;
      v35 = v30;
      v36 = &v34;
      v37 = ".neg";
      v36 = (const char **)sub_19FE020((__int64 *)a1, (__int64)&v36, a2, a2);
      v14 = v36;
      sub_1A062A0(a3, &v36);
      return (__int64)v14;
    }
    while ( 1 )
    {
      v14 = sub_1648700(v24);
      if ( sub_15FB6B0((__int64)v14, v12, v25, v26) || (v12 = 0, sub_15FB6D0((__int64)v14, 0, v27, v28)) )
      {
        v29 = *(_QWORD *)(v14[5] + 56LL);
        if ( *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL) == v29 )
          break;
      }
      v24 = *(_QWORD *)(v24 + 8);
      if ( !v24 )
        goto LABEL_30;
    }
    v31 = *(_BYTE *)(a1 + 16);
    if ( v31 <= 0x17u )
    {
      v33 = *(_QWORD *)(v29 + 80);
      if ( !v33 )
        BUG();
      v32 = *(_QWORD *)(v33 + 24);
      if ( !v32 )
        goto LABEL_38;
    }
    else
    {
      if ( v31 == 29 )
        v32 = *(_QWORD *)(*(_QWORD *)(a1 - 48) + 48LL);
      else
        v32 = *(_QWORD *)(a1 + 32);
      while ( 1 )
      {
        if ( !v32 )
          BUG();
        if ( *(_BYTE *)(v32 - 8) != 77 )
          break;
        v32 = *(_QWORD *)(v32 + 8);
      }
    }
    v32 -= 24;
LABEL_38:
    sub_15F22F0(v14, v32);
    if ( *((_BYTE *)v14 + 16) == 37 )
    {
      sub_15F2310((__int64)v14, 0);
      sub_15F2330((__int64)v14, 0);
    }
    else
    {
      sub_15F2780((unsigned __int8 *)v14, a2);
    }
    v36 = (const char **)v14;
    sub_1A062A0(a3, &v36);
    return (__int64)v14;
  }
  v7 = *(_QWORD *)a1;
  v8 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
  if ( v8 == 16 )
    v8 = *(_BYTE *)(**(_QWORD **)(v7 + 16) + 8LL);
  if ( (unsigned __int8)(v8 - 1) > 5u )
    return sub_15A2B90((__int64 *)a1, 0, 0, a4, a5, a6, a7);
  else
    return sub_15A2BF0((__int64 *)a1, a2, v7, a4, a5, a6, a7);
}
