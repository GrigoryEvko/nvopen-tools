// Function: sub_22B6280
// Address: 0x22b6280
//
void __fastcall sub_22B6280(__int64 a1, __int64 a2, char **a3, char **a4)
{
  unsigned int *v4; // r13
  __int64 v5; // r8
  unsigned __int64 v6; // r9
  _QWORD *v7; // r12
  _QWORD *v8; // rdi
  _QWORD *v9; // rbx
  _QWORD *v10; // r12
  __int64 v11; // rsi
  unsigned __int64 *v12; // rax
  unsigned __int64 *v13; // rcx
  unsigned __int64 v14; // rsi
  __int64 v15; // rdx
  _QWORD *i; // [rsp+20h] [rbp-90h]
  _QWORD *v20; // [rsp+28h] [rbp-88h]
  _QWORD *v21; // [rsp+30h] [rbp-80h] BYREF
  __int16 v22; // [rsp+38h] [rbp-78h]
  char *v23; // [rsp+40h] [rbp-70h] BYREF
  char *v24; // [rsp+48h] [rbp-68h]
  __int64 v25; // [rsp+50h] [rbp-60h]
  char *v26[10]; // [rsp+60h] [rbp-50h] BYREF

  v4 = (unsigned int *)(a1 + 192);
  v23 = 0;
  v24 = 0;
  v25 = 0;
  memset(v26, 0, 24);
  sub_22B3D40(a1 + 192, a2);
  v7 = *(_QWORD **)(a2 + 32);
  for ( i = (_QWORD *)(a2 + 24); i != v7; v7 = (_QWORD *)v7[1] )
  {
    if ( !v7 )
      BUG();
    v8 = v7 + 2;
    if ( v7 + 2 != (_QWORD *)(v7[2] & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v9 = (_QWORD *)v7[3];
      if ( v8 == v9 )
        goto LABEL_10;
      v20 = v7;
      v10 = (_QWORD *)v7[3];
      do
      {
        v11 = (__int64)(v10 - 3);
        if ( !v10 )
          v11 = 0;
        sub_22B6090((__int64)v4, v11, &v23, v26, v5, v6);
        v10 = (_QWORD *)v10[1];
      }
      while ( v8 != v10 );
      v7 = v20;
      v9 = (_QWORD *)v20[3];
      if ( v9 )
LABEL_10:
        v9 -= 3;
      v21 = v9 + 6;
      v22 = 0;
      sub_22B4880(v4, (__int64 *)&v21, (__int64)v26, (__int64)&v23, 1, v6);
      if ( v24 != v23 )
      {
        v12 = (unsigned __int64 *)*((_QWORD *)v24 - 1);
        v13 = *(unsigned __int64 **)(a1 + 288);
        v14 = *v13;
        v15 = *v12 & 7;
        v12[1] = (unsigned __int64)v13;
        v14 &= 0xFFFFFFFFFFFFFFF8LL;
        *v12 = v14 | v15;
        *(_QWORD *)(v14 + 8) = v12;
        *v13 = *v13 & 7 | (unsigned __int64)v12;
      }
    }
  }
  sub_22B0250(a3, (__int64)&v23);
  sub_22B0470(a4, (__int64)v26);
  if ( v26[0] )
    j_j___libc_free_0((unsigned __int64)v26[0]);
  if ( v23 )
    j_j___libc_free_0((unsigned __int64)v23);
}
