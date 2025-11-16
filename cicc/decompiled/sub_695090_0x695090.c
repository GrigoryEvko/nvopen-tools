// Function: sub_695090
// Address: 0x695090
//
__int64 __fastcall sub_695090(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rax
  unsigned int *v5; // rdi
  _BOOL4 v6; // r12d
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 *v9; // r14
  int v10; // eax
  __int64 v11; // rax
  _BYTE *v12; // r8
  __int64 v13; // rsi
  unsigned int v14; // r12d
  __int64 v15; // rdi
  _QWORD *v18; // [rsp+18h] [rbp-528h]
  __int64 v19; // [rsp+20h] [rbp-520h]
  unsigned int v21; // [rsp+34h] [rbp-50Ch] BYREF
  __int64 v22; // [rsp+38h] [rbp-508h] BYREF
  __int64 v23; // [rsp+40h] [rbp-500h] BYREF
  __int64 i; // [rsp+48h] [rbp-4F8h] BYREF
  _BYTE v25[160]; // [rsp+50h] [rbp-4F0h] BYREF
  _BYTE v26[68]; // [rsp+F0h] [rbp-450h] BYREF
  __int64 v27; // [rsp+134h] [rbp-40Ch]
  _BYTE v28[144]; // [rsp+180h] [rbp-3C0h] BYREF
  __int64 v29; // [rsp+210h] [rbp-330h]
  _BYTE v30[352]; // [rsp+250h] [rbp-2F0h] BYREF
  _BYTE v31[400]; // [rsp+3B0h] [rbp-190h] BYREF

  v18 = *(_QWORD **)(a1 + 8);
  v4 = v18[19];
  v22 = 0;
  v23 = 0;
  for ( i = 0; *(_BYTE *)(v4 + 140) == 12; v4 = *(_QWORD *)(v4 + 160) )
    ;
  v5 = &v21;
  v19 = *(_QWORD *)(*(_QWORD *)(v4 + 168) + 40LL);
  v6 = v19 != 0;
  sub_7296C0(&v21);
  v7 = *a2 + 24 * a2[2];
  if ( v7 != *a2 )
  {
    v8 = *a2;
    v9 = &v22;
    v10 = 1;
    do
    {
      if ( (v10 & v6) != 0 )
      {
        if ( *(_BYTE *)v8 != 2 )
          goto LABEL_7;
        sub_6E6A50(*(_QWORD *)(v8 + 8), v31);
        v5 = (unsigned int *)v31;
        sub_82F1E0(v31, 0, v31);
      }
      else
      {
        v5 = 0;
        v11 = sub_6E2F40(0);
        *v9 = v11;
        v9 = (__int64 *)v11;
        if ( *(_BYTE *)v8 != 2 )
LABEL_7:
          sub_721090(v5);
        v5 = *(unsigned int **)(v8 + 8);
        sub_6E6A50(v5, *(_QWORD *)(v11 + 24) + 8LL);
      }
      v8 += 24;
      v10 = 0;
    }
    while ( v7 != v8 );
  }
  sub_6E1E00(4, v25, 0, 0);
  LODWORD(v12) = v19;
  if ( v19 )
    v12 = v31;
  v13 = 0;
  v14 = sub_84C4B0(
          *v18,
          0,
          0,
          v6,
          (_DWORD)v12,
          (unsigned int)&v22,
          0,
          0,
          0,
          0,
          0,
          1,
          0,
          0,
          (__int64)a3,
          dword_4F06650[0],
          0,
          0,
          (__int64)v30,
          (__int64)&v23);
  if ( v14 )
  {
    v14 = 0;
    sub_7022F0(
      (unsigned int)v30,
      (unsigned int)v31,
      v23,
      1,
      1,
      1,
      0,
      0,
      (__int64)a3,
      (__int64)a3,
      (__int64)a3,
      (__int64)v26,
      0,
      (__int64)&i);
    v13 = 1;
    v27 = *a3;
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 8u;
    sub_6E6B60(v26, 1);
    if ( v26[16] == 2 )
    {
      v13 = a4;
      v29 = 0;
      v14 = 1;
      sub_72A510(v28, a4);
    }
    sub_6E4710(v26);
  }
  v15 = v22;
  sub_6E1990(v22);
  sub_6E2B30(v15, v13);
  sub_729730(v21);
  return v14;
}
