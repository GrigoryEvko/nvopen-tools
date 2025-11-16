// Function: sub_810650
// Address: 0x810650
//
void __fastcall sub_810650(__int64 a1, int a2, _QWORD *a3)
{
  bool v5; // zf
  __int64 v6; // rax
  __int64 v7; // r14
  char v8; // dl
  char v9; // al
  char *v10; // rdi
  _QWORD *v11; // rsi
  const char *v12; // r14
  size_t v13; // rax
  __int64 v14; // rdx
  size_t v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // rax
  int v21; // [rsp+4h] [rbp-6Ch] BYREF
  __int64 *v22; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v23[12]; // [rsp+10h] [rbp-60h] BYREF

  v5 = *(_BYTE *)(a1 + 140) == 12;
  v21 = 0;
  if ( v5 )
    goto LABEL_15;
  if ( a2 && (unsigned int)sub_80C5A0(a1, 6, 0, 0, v23, a3) )
    return;
  v6 = sub_809820(a1);
  v7 = v6;
  if ( !v6 )
  {
LABEL_15:
    sub_811730(a1, 6, &v21, &v22, 0, a3);
    v8 = *(_BYTE *)(a1 + 140);
    if ( (unsigned __int8)(v8 - 9) > 2u )
      goto LABEL_9;
    goto LABEL_16;
  }
  if ( !(unsigned int)sub_80C5A0(v6, 59, 0, 1, v23, a3) )
  {
    sub_811730(a1, 6, &v21, &v22, 0, a3);
    if ( !a3[5] )
      sub_80A250(v7, 59, 0, (__int64)a3);
    v8 = *(_BYTE *)(a1 + 140);
    if ( (unsigned __int8)(v8 - 9) > 2u )
    {
LABEL_9:
      v9 = *(_BYTE *)(a1 + 89) & 8;
      if ( v8 == 12 && *(_BYTE *)(a1 + 184) == 10 )
      {
        if ( v9 )
          v12 = *(const char **)(a1 + 24);
        else
          v12 = *(const char **)(a1 + 8);
        if ( !a3[5] )
          sub_80A250(a1, 6, 1, (__int64)a3);
        v13 = strlen(v12);
        if ( v13 > 9 )
        {
          v14 = (int)sub_622470(v13, v23);
        }
        else
        {
          v14 = 1;
          LOWORD(v23[0]) = (unsigned __int8)(v13 + 48);
        }
        *a3 += v14;
        sub_8238B0(qword_4F18BE0, v23, v14);
        v15 = strlen(v12);
        *a3 += v15;
        sub_8238B0(qword_4F18BE0, v12, v15);
        v23[0] = **(_QWORD **)(a1 + 168);
        sub_811CB0(v23, 0, 0, a3);
        goto LABEL_13;
      }
      if ( v9 )
      {
        v10 = *(char **)(a1 + 24);
        v11 = a3;
        if ( v10 )
        {
LABEL_12:
          sub_80BC40(v10, v11);
LABEL_13:
          sub_80C110(v21, v22, a3);
          return;
        }
      }
      else
      {
        v10 = *(char **)(a1 + 8);
        v11 = a3;
        if ( v10 )
          goto LABEL_12;
      }
      sub_80FE00(a1, (__int64)v11);
      goto LABEL_13;
    }
LABEL_16:
    sub_813620(a1, a3);
    goto LABEL_13;
  }
  v16 = *(_QWORD *)(v7 + 40);
  if ( (*(_BYTE *)(v7 + 89) & 4) != 0 )
  {
    if ( !v16 || *(_BYTE *)(v16 + 28) != 3 )
      goto LABEL_31;
  }
  else if ( !v16 || *(_BYTE *)(v16 + 28) != 3 )
  {
    goto LABEL_36;
  }
  if ( (*(_BYTE *)(*(_QWORD *)(v16 + 32) + 124LL) & 0x10) != 0 )
  {
LABEL_36:
    sub_80C5A0(v7, 59, 0, 0, v23, a3);
    v23[0] = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 168LL);
    sub_811CB0(v23, 0, 0, a3);
    return;
  }
LABEL_31:
  v17 = (_QWORD *)qword_4F18BE0;
  ++*a3;
  v18 = v17[2];
  if ( (unsigned __int64)(v18 + 1) > v17[1] )
  {
    sub_823810(v17);
    v17 = (_QWORD *)qword_4F18BE0;
    v18 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v17[4] + v18) = 78;
  ++v17[2];
  sub_80C5A0(v7, 59, 0, 0, v23, a3);
  v23[0] = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 168LL);
  sub_811CB0(v23, 0, 0, a3);
  v19 = (_QWORD *)qword_4F18BE0;
  ++*a3;
  v20 = v19[2];
  if ( (unsigned __int64)(v20 + 1) > v19[1] )
  {
    sub_823810(v19);
    v19 = (_QWORD *)qword_4F18BE0;
    v20 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v19[4] + v20) = 69;
  ++v19[2];
}
