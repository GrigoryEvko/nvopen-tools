// Function: sub_7D4C80
// Address: 0x7d4c80
//
__int64 __fastcall sub_7D4C80(__int64 a1, _QWORD **a2, _QWORD *a3, _QWORD **a4, _QWORD **a5, unsigned int a6)
{
  _QWORD *v6; // r14
  __int64 v7; // r13
  unsigned __int8 v8; // al
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rdi
  __int64 v15; // r15
  _QWORD *i; // r15
  __int64 *v17; // rax
  char v18; // dl
  _QWORD *v19; // rbx
  _QWORD *v20; // r14
  __int64 v21; // r13
  _QWORD *v22; // r12
  __int64 *v23; // r15
  __int64 v24; // r15
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  _QWORD *v27; // [rsp+0h] [rbp-90h]
  _QWORD *v30; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+30h] [rbp-60h]
  _QWORD *v34; // [rsp+38h] [rbp-58h]
  __int64 *v35; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v36; // [rsp+48h] [rbp-48h] BYREF
  _QWORD *v37; // [rsp+50h] [rbp-40h] BYREF
  __int64 *v38; // [rsp+58h] [rbp-38h] BYREF

  v6 = (_QWORD *)*a3;
  v33 = (__int64)a2;
  v36 = *a5;
  v37 = *a4;
  v30 = a4;
  v35 = 0;
  if ( v6 )
  {
    while ( 1 )
    {
      v7 = v6[1];
      v8 = *(_BYTE *)(v7 + 140);
      if ( v8 <= 0xBu )
        break;
      if ( v8 == 20 && qword_4D049B0 )
      {
        a2 = &v37;
        sub_7CEBB0(*(_QWORD *)(qword_4D049B0 + 88LL), &v37);
        v6 = (_QWORD *)*v6;
        if ( !v6 )
          goto LABEL_13;
      }
      else
      {
LABEL_8:
        v6 = (_QWORD *)*v6;
        if ( !v6 )
          goto LABEL_13;
      }
    }
    if ( v8 <= 8u )
    {
      if ( v8 != 2 || (*(_BYTE *)(v7 + 161) & 8) == 0 )
        goto LABEL_8;
      goto LABEL_6;
    }
    v15 = *(_QWORD *)(v7 + 168);
    if ( unk_4D045F0 | unk_4F06A80 | unk_4F06A7C | unk_4F06A78
      || (a2 = (_QWORD **)dword_4D045F4, dword_4D045F4)
      && ((a4 = (_QWORD **)dword_4D0455C, !dword_4D0455C) || unk_4D04600 > 0x30DA3u) )
    {
      if ( (*(_BYTE *)(v15 + 110) & 0x40) != 0 )
        goto LABEL_8;
    }
    sub_7CED60(v6[1], &v37, &v36);
    if ( (!dword_4F077BC || qword_4F077A8 > 0x9E33u)
      && dword_4F077C4 == 2
      && (unsigned int)sub_8D23B0(v7)
      && (unsigned int)sub_8D3A70(v7) )
    {
      sub_8AD220(v7, 0);
    }
    for ( i = *(_QWORD **)v15; i; i = (_QWORD *)*i )
      sub_7CED60(i[5], &v37, &v36);
    if ( (*(_BYTE *)(v7 + 177) & 0x10) == 0 || (v17 = *(__int64 **)(*(_QWORD *)(v7 + 168) + 168LL), (v38 = v17) == 0) )
    {
LABEL_6:
      a2 = &v37;
      if ( (*(_BYTE *)(v7 + 89) & 4) != 0 )
        sub_7CED60(*(_QWORD *)(*(_QWORD *)(v7 + 40) + 32LL), &v37, &v36);
      else
        sub_7CECA0(v7, &v37);
      goto LABEL_8;
    }
    v18 = *((_BYTE *)v17 + 8);
    if ( v18 != 3 )
      goto LABEL_36;
    while ( 1 )
    {
      sub_72F220(&v38);
      v17 = v38;
      if ( !v38 )
        goto LABEL_6;
      if ( *((_BYTE *)v38 + 8) == 2 )
LABEL_37:
        sub_7CF470(v17[4], &v37, &v36);
      while ( 1 )
      {
        v17 = (__int64 *)*v38;
        v38 = v17;
        if ( !v17 )
          goto LABEL_6;
        v18 = *((_BYTE *)v17 + 8);
        if ( v18 == 3 )
          break;
LABEL_36:
        if ( v18 == 2 )
          goto LABEL_37;
      }
    }
  }
LABEL_13:
  v9 = a6;
  if ( a6 )
  {
    a2 = &v37;
    v9 = qword_4D049B8[11];
    sub_7CEBB0(v9, &v37);
  }
  v34 = v36;
  if ( unk_4D03FC8 <= 0
    || unk_4F04C48 == -1
    || (*(_BYTE *)(qword_4F04C68[0] + 10LL) & 4) == 0
    || dword_4F04C44 != -1
    || (v10 = 776LL * dword_4F04C64, (*(_BYTE *)(qword_4F04C68[0] + v10 + 6) & 2) != 0) )
  {
    v9 = v33;
    a2 = (_QWORD **)v37;
    sub_7D4B20(v33, v37, v36, qword_4D03FF0, (__int64 *)&v35);
  }
  else
  {
    v19 = (_QWORD *)unk_4D03FF8;
    if ( unk_4D03FF8 )
    {
      v27 = v37;
      while ( 1 )
      {
        v20 = v27;
        v21 = v19[1];
        v22 = 0;
        if ( v27 )
          break;
LABEL_61:
        v26 = 0;
        a2 = (_QWORD **)v22;
        if ( v21 == qword_4D03FF0 )
          v26 = v34;
        sub_7D4B20(v33, v22, v26, v21, (__int64 *)&v35);
        v9 = (__int64)v22;
        sub_8788C0(v22);
        v19 = (_QWORD *)*v19;
        if ( !v19 )
          goto LABEL_20;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v23 = (__int64 *)v20[1];
          if ( v23 )
            break;
LABEL_60:
          v25 = (_QWORD *)sub_8787C0();
          *v25 = v22;
          v22 = v25;
          v25[1] = v23;
          v20 = (_QWORD *)*v20;
          if ( !v20 )
            goto LABEL_61;
        }
        v24 = *v23;
        if ( (unsigned int)sub_879530(v24) )
        {
          if ( v21 != sub_880F80(v24) )
            goto LABEL_54;
        }
        else
        {
          v24 = sub_8CFEE0(v24, v21);
        }
        if ( v24 )
        {
          v23 = *(__int64 **)(v24 + 88);
          goto LABEL_60;
        }
LABEL_54:
        v20 = (_QWORD *)*v20;
        if ( !v20 )
          goto LABEL_61;
      }
    }
  }
LABEL_20:
  if ( a1 )
  {
    v11 = (__int64 *)sub_878440(v9, a2, v10, a4);
    v12 = (__int64)v35;
    v11[1] = a1;
    *v11 = v12;
    v35 = v11;
  }
  sub_878510(*a3);
  *a3 = 0;
  sub_8788C0(v37);
  v13 = v36;
  *v30 = 0;
  sub_878510(v13);
  *a5 = 0;
  return (__int64)v35;
}
