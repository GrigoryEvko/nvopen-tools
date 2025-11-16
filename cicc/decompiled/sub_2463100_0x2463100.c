// Function: sub_2463100
// Address: 0x2463100
//
_BYTE *__fastcall sub_2463100(unsigned __int8 *a1)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rdi
  const char *v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  _BYTE *v10; // rax
  size_t v11; // r14
  _QWORD *v12; // rax
  _DWORD *v13; // rdx
  char *v14; // rax
  unsigned __int8 *v15; // r14
  size_t v16; // rax
  size_t v17; // r15
  _BYTE *v18; // rax
  _QWORD *v19; // rax
  _DWORD *v20; // rdx
  __int64 v21; // r12
  _BYTE *result; // rax

  if ( *a1 == 85 )
  {
    v2 = sub_CB72A0();
    v3 = v2[4];
    v4 = (__int64)v2;
    if ( (unsigned __int64)(v2[3] - v3) <= 8 )
    {
      v4 = sub_CB6200((__int64)v2, "ZZZ call ", 9u);
    }
    else
    {
      *(_BYTE *)(v3 + 8) = 32;
      *(_QWORD *)v3 = 0x6C6C6163205A5A5ALL;
      v2[4] += 9LL;
    }
    v5 = *((_QWORD *)a1 - 4);
    if ( v5 )
    {
      if ( *(_BYTE *)v5 )
      {
        v5 = 0;
      }
      else if ( *(_QWORD *)(v5 + 24) != *((_QWORD *)a1 + 10) )
      {
        v5 = 0;
      }
    }
    v6 = sub_BD5D20(v5);
    v8 = *(_BYTE **)(v4 + 32);
    v9 = (unsigned __int8 *)v6;
    v10 = *(_BYTE **)(v4 + 24);
    v11 = v7;
    if ( v10 - v8 < v7 )
    {
      v4 = sub_CB6200(v4, v9, v7);
      v10 = *(_BYTE **)(v4 + 24);
      v8 = *(_BYTE **)(v4 + 32);
    }
    else if ( v7 )
    {
      memcpy(v8, v9, v7);
      v10 = *(_BYTE **)(v4 + 24);
      v8 = (_BYTE *)(v11 + *(_QWORD *)(v4 + 32));
      *(_QWORD *)(v4 + 32) = v8;
    }
    if ( v8 == v10 )
      goto LABEL_12;
    goto LABEL_22;
  }
  v12 = sub_CB72A0();
  v13 = (_DWORD *)v12[4];
  v4 = (__int64)v12;
  if ( v12[3] - (_QWORD)v13 <= 3u )
  {
    v4 = sub_CB6200((__int64)v12, "ZZZ ", 4u);
  }
  else
  {
    *v13 = 542792282;
    v12[4] += 4LL;
  }
  v14 = sub_B458E0((unsigned int)*a1 - 29);
  v15 = (unsigned __int8 *)v14;
  if ( !v14 )
    goto LABEL_20;
  v16 = strlen(v14);
  v8 = *(_BYTE **)(v4 + 32);
  v17 = v16;
  v18 = *(_BYTE **)(v4 + 24);
  if ( v17 > v18 - v8 )
  {
    v4 = sub_CB6200(v4, v15, v17);
LABEL_20:
    v18 = *(_BYTE **)(v4 + 24);
    v8 = *(_BYTE **)(v4 + 32);
    goto LABEL_21;
  }
  if ( v17 )
  {
    memcpy(v8, v15, v17);
    v18 = *(_BYTE **)(v4 + 24);
    v8 = (_BYTE *)(v17 + *(_QWORD *)(v4 + 32));
    *(_QWORD *)(v4 + 32) = v8;
  }
LABEL_21:
  if ( v18 == v8 )
  {
LABEL_12:
    sub_CB6200(v4, (unsigned __int8 *)"\n", 1u);
    goto LABEL_23;
  }
LABEL_22:
  *v8 = 10;
  ++*(_QWORD *)(v4 + 32);
LABEL_23:
  v19 = sub_CB72A0();
  v20 = (_DWORD *)v19[4];
  v21 = (__int64)v19;
  if ( v19[3] - (_QWORD)v20 <= 3u )
  {
    v21 = sub_CB6200((__int64)v19, "QQQ ", 4u);
  }
  else
  {
    *v20 = 542200145;
    v19[4] += 4LL;
  }
  sub_A69870((__int64)a1, (_BYTE *)v21, 0);
  result = *(_BYTE **)(v21 + 32);
  if ( *(_BYTE **)(v21 + 24) == result )
    return (_BYTE *)sub_CB6200(v21, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(v21 + 32);
  return result;
}
