// Function: sub_82E650
// Address: 0x82e650
//
void __fastcall sub_82E650(__int64 *a1, _QWORD *a2, __int64 a3, unsigned __int8 a4, _QWORD *a5)
{
  __int64 *v5; // r15
  __int64 v10; // rdx
  char v11; // al
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rdx
  int v15; // esi
  char *v16; // rdx
  char v17; // di
  char *v18; // rax
  const char *v19; // rsi
  char *v20; // rcx
  char *v21; // rdx
  char *v22; // rax
  char *v23; // rax
  const char *v24; // [rsp+0h] [rbp-C0h]
  char *v25; // [rsp+8h] [rbp-B8h]
  char *v26; // [rsp+8h] [rbp-B8h]
  char *v27; // [rsp+8h] [rbp-B8h]
  char *v28; // [rsp+10h] [rbp-B0h]
  char *v29; // [rsp+10h] [rbp-B0h]
  char *v30; // [rsp+10h] [rbp-B0h]
  char *v31; // [rsp+10h] [rbp-B0h]
  int i; // [rsp+1Ch] [rbp-A4h]
  char s[160]; // [rsp+20h] [rbp-A0h] BYREF

  v5 = a1;
  for ( i = a4; v5; v5 = (__int64 *)*v5 )
  {
    v10 = v5[1];
    if ( v10 )
    {
      v11 = *(_BYTE *)(v10 + 80);
      v12 = v5[1];
      if ( v11 == 16 )
      {
        v12 = **(_QWORD **)(v10 + 88);
        v11 = *(_BYTE *)(v12 + 80);
      }
      v13 = v12;
      if ( v11 == 24 )
        v13 = *(_QWORD *)(v12 + 88);
      if ( (*(_BYTE *)(v10 + 82) & 4) != 0 )
      {
        v15 = 423;
      }
      else
      {
        v14 = *(_QWORD *)(v13 + 88);
        if ( *(_BYTE *)(v13 + 80) == 20 )
          v14 = *(_QWORD *)(v14 + 176);
        if ( (*(_BYTE *)(v14 + 194) & 0x40) != 0 )
        {
          sub_67E270(a5, 3092, (_QWORD *)(*(_QWORD *)(v14 + 360) + 8LL), **(_QWORD **)(v14 + 232));
          continue;
        }
        v10 = v12;
        v15 = (*((_BYTE *)v5 + 145) & 2) == 0 ? 421 : 3025;
        if ( v11 == 24 )
          v10 = *(_QWORD *)(v12 + 88);
      }
    }
    else
    {
      v10 = v5[7];
      if ( !v10 )
      {
        v16 = (char *)v5[6];
        v17 = v16[1];
        if ( v17 == 59 || !v17 )
        {
          v29 = (char *)qword_4F064C0[i];
          v19 = "%s %s";
          v20 = sub_827390(*v16);
          v21 = v29;
        }
        else if ( a4 == 43 )
        {
          v27 = (char *)v5[6];
          v31 = sub_827390(v17);
          v23 = sub_827390(*v27);
          v20 = v31;
          v19 = "%s[%s]";
          v21 = v23;
        }
        else
        {
          if ( a4 != 44 )
          {
            v24 = (const char *)qword_4F064C0[i];
            v25 = (char *)v5[6];
            v28 = sub_827390(v17);
            v18 = sub_827390(*v25);
            sprintf(s, "%s %s %s", v18, v24, v28);
LABEL_31:
            sub_67DCF0(a5, 422, (__int64)s);
            continue;
          }
          v26 = (char *)v5[6];
          v30 = sub_827390(v17);
          v22 = sub_827390(*v26);
          v20 = v30;
          v19 = "expression ? %s : %s";
          v21 = v22;
        }
        sprintf(s, v19, v21, v20);
        goto LABEL_31;
      }
      v15 = 981;
    }
    sub_67E1D0(a5, v15, v10);
  }
  if ( a3 )
  {
    if ( a4 )
    {
      sub_82B170(a3, a4, a5);
    }
    else
    {
      if ( a2 )
        a2 = (_QWORD *)*a2;
      sub_82E4F0((__int64)a2, a3, a5);
    }
  }
}
