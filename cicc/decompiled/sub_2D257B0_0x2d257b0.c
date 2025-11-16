// Function: sub_2D257B0
// Address: 0x2d257b0
//
__int64 __fastcall sub_2D257B0(__int64 a1, unsigned int *a2)
{
  __int64 v2; // rdx
  __int64 v3; // r12
  _BYTE *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rax
  _QWORD *i; // r13
  __int64 v9; // r15
  _QWORD *v10; // rax
  _QWORD *v11; // rbx
  int v12; // r15d
  __int64 v13; // r14
  __int64 v14; // r12
  const char *v15; // rax
  size_t v16; // rdx
  _BYTE *v17; // rdi
  unsigned __int8 *v18; // rsi
  _BYTE *v19; // rax
  _QWORD *v20; // rax
  _WORD *v21; // rdx
  __int64 v23; // rax
  size_t v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h] BYREF
  _QWORD *v26; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v2) <= 8 )
  {
    a1 = sub_CB6200(a1, "DEF Var=[", 9u);
  }
  else
  {
    *(_BYTE *)(v2 + 8) = 91;
    *(_QWORD *)v2 = 0x3D72615620464544LL;
    *(_QWORD *)(a1 + 32) += 9LL;
  }
  v3 = sub_CB59D0(a1, *a2);
  v4 = *(_BYTE **)(v3 + 32);
  if ( *(_BYTE **)(v3 + 24) == v4 )
  {
    v23 = sub_CB6200(v3, (unsigned __int8 *)"]", 1u);
    v5 = *(_QWORD *)(v23 + 32);
    v3 = v23;
  }
  else
  {
    *v4 = 93;
    v5 = *(_QWORD *)(v3 + 32) + 1LL;
    *(_QWORD *)(v3 + 32) = v5;
  }
  if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v5) <= 5 )
  {
    v3 = sub_CB6200(v3, " Expr=", 6u);
  }
  else
  {
    *(_DWORD *)v5 = 1886930208;
    *(_WORD *)(v5 + 4) = 15730;
    *(_QWORD *)(v3 + 32) += 6LL;
  }
  sub_A61DE0(*((const char **)a2 + 1), v3, 0);
  v6 = *(_QWORD *)(v3 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v6) <= 8 )
  {
    sub_CB6200(v3, " Values=(", 9u);
  }
  else
  {
    *(_BYTE *)(v6 + 8) = 40;
    *(_QWORD *)v6 = 0x3D7365756C615620LL;
    *(_QWORD *)(v3 + 32) += 9LL;
  }
  sub_B58DC0(&v25, (unsigned __int8 **)a2 + 3);
  v7 = v25;
  for ( i = v26; (_QWORD *)v7 != i; v7 = (unsigned __int64)(v11 + 1) | 4 )
  {
    while ( 1 )
    {
      v9 = v7;
      v10 = (_QWORD *)(v7 & 0xFFFFFFFFFFFFFFF8LL);
      v11 = v10;
      v12 = (v9 >> 2) & 1;
      if ( v12 )
        v10 = (_QWORD *)*v10;
      v13 = v10[17];
      v14 = (__int64)sub_CB72A0();
      v15 = sub_BD5D20(v13);
      v17 = *(_BYTE **)(v14 + 32);
      v18 = (unsigned __int8 *)v15;
      v19 = *(_BYTE **)(v14 + 24);
      if ( v19 - v17 < v16 )
      {
        v14 = sub_CB6200(v14, v18, v16);
        v19 = *(_BYTE **)(v14 + 24);
        v17 = *(_BYTE **)(v14 + 32);
      }
      else if ( v16 )
      {
        v24 = v16;
        memcpy(v17, v18, v16);
        v19 = *(_BYTE **)(v14 + 24);
        v17 = (_BYTE *)(v24 + *(_QWORD *)(v14 + 32));
        *(_QWORD *)(v14 + 32) = v17;
      }
      if ( v19 == v17 )
      {
        sub_CB6200(v14, (unsigned __int8 *)" ", 1u);
      }
      else
      {
        *v17 = 32;
        ++*(_QWORD *)(v14 + 32);
      }
      if ( v12 )
        break;
      v7 = (__int64)(v11 + 18);
      if ( v11 + 18 == i )
        goto LABEL_21;
    }
  }
LABEL_21:
  v20 = sub_CB72A0();
  v21 = (_WORD *)v20[4];
  if ( v20[3] - (_QWORD)v21 <= 1u )
    return sub_CB6200((__int64)v20, (unsigned __int8 *)")\n", 2u);
  *v21 = 2601;
  v20[4] += 2LL;
  return 2601;
}
