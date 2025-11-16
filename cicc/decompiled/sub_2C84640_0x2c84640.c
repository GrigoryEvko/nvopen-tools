// Function: sub_2C84640
// Address: 0x2c84640
//
__int64 __fastcall sub_2C84640(_QWORD *a1, __int64 a2, char a3)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // r13
  _QWORD *v6; // r15
  _QWORD *v7; // r12
  bool v8; // zf
  __int64 v9; // rdi
  char v10; // r9
  _QWORD *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  char v16; // r12
  _QWORD *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rdi
  _QWORD *v22; // r12
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v26; // [rsp+10h] [rbp-80h]
  _QWORD *v27; // [rsp+18h] [rbp-78h]
  _QWORD *v28; // [rsp+20h] [rbp-70h]
  __int64 v29; // [rsp+28h] [rbp-68h]
  _QWORD *v31; // [rsp+38h] [rbp-58h]
  char v32; // [rsp+38h] [rbp-58h]
  char v33; // [rsp+4Dh] [rbp-43h] BYREF
  char v34; // [rsp+4Eh] [rbp-42h] BYREF
  char v35; // [rsp+4Fh] [rbp-41h] BYREF
  unsigned __int64 v36; // [rsp+50h] [rbp-40h] BYREF
  unsigned __int64 *v37[7]; // [rsp+58h] [rbp-38h] BYREF

  v3 = a2 + 72;
  result = *(_QWORD *)(v3 + 8);
  v28 = a1 + 10;
  v29 = result;
  v26 = v3;
  v27 = a1 + 4;
  if ( result != v3 )
  {
    while ( 1 )
    {
      v33 = 0;
      v34 = 0;
      v5 = v29 - 24;
      if ( !v29 )
        v5 = 0;
      v36 = v5;
      v6 = *(_QWORD **)(v5 + 56);
      v31 = (_QWORD *)(v5 + 48);
      if ( a3 )
        break;
      if ( v6 == v31 )
        goto LABEL_10;
      while ( 1 )
      {
        v21 = (__int64)(v6 - 3);
        if ( !v6 )
          v21 = 0;
        if ( sub_2C83D20(v21) )
          break;
        v6 = (_QWORD *)v6[1];
        if ( v6 == v31 )
          goto LABEL_10;
      }
      v22 = *(_QWORD **)(v5 + 56);
      if ( v22 == v6 )
      {
LABEL_47:
        v10 = v33;
        goto LABEL_11;
      }
      do
      {
        v35 = 0;
        LOBYTE(v37[0]) = 0;
        v8 = (*v6 & 0xFFFFFFFFFFFFFFF8LL) == 0;
        v23 = (*v6 & 0xFFFFFFFFFFFFFFF8LL) - 24;
        v6 = (_QWORD *)(*v6 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v8 )
          v23 = 0;
        sub_2C83AE0(v23, &v35, v37);
        v10 = v35 | v33;
        v33 |= v35;
        v34 |= LOBYTE(v37[0]);
      }
      while ( v22 != v6 );
      v11 = (_QWORD *)a1[5];
      if ( !v11 )
      {
LABEL_40:
        v12 = (__int64)v27;
LABEL_18:
        v32 = v10;
        v37[0] = &v36;
        v15 = sub_2C84590(a1 + 3, v12, v37);
        v10 = v32;
        v12 = v15;
        goto LABEL_19;
      }
LABEL_12:
      v12 = (__int64)v27;
      do
      {
        while ( 1 )
        {
          v13 = v11[2];
          v14 = v11[3];
          if ( v11[4] >= v36 )
            break;
          v11 = (_QWORD *)v11[3];
          if ( !v14 )
            goto LABEL_16;
        }
        v12 = (__int64)v11;
        v11 = (_QWORD *)v11[2];
      }
      while ( v13 );
LABEL_16:
      if ( v27 == (_QWORD *)v12 || *(_QWORD *)(v12 + 32) > v36 )
        goto LABEL_18;
LABEL_19:
      v16 = v34;
      *(_BYTE *)(v12 + 40) = v10;
      v17 = (_QWORD *)a1[11];
      if ( v17 )
      {
        v18 = (__int64)v28;
        do
        {
          while ( 1 )
          {
            v19 = v17[2];
            v20 = v17[3];
            if ( v17[4] >= v36 )
              break;
            v17 = (_QWORD *)v17[3];
            if ( !v20 )
              goto LABEL_24;
          }
          v18 = (__int64)v17;
          v17 = (_QWORD *)v17[2];
        }
        while ( v19 );
LABEL_24:
        if ( v28 != (_QWORD *)v18 && *(_QWORD *)(v18 + 32) <= v36 )
          goto LABEL_27;
      }
      else
      {
        v18 = (__int64)v28;
      }
      v37[0] = &v36;
      v18 = sub_2C84590(a1 + 9, v18, v37);
LABEL_27:
      *(_BYTE *)(v18 + 40) = v16;
      result = *(_QWORD *)(v29 + 8);
      v29 = result;
      if ( v26 == result )
        return result;
    }
    v7 = (_QWORD *)(v5 + 48);
    while ( v7 != v6 )
    {
      v8 = (*v7 & 0xFFFFFFFFFFFFFFF8LL) == 0;
      v9 = (*v7 & 0xFFFFFFFFFFFFFFF8LL) - 24;
      v7 = (_QWORD *)(*v7 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v8 )
        v9 = 0;
      if ( sub_2C83D20(v9) )
      {
        if ( v7 == v31 )
          goto LABEL_47;
        do
        {
          v24 = (__int64)(v7 - 3);
          if ( !v7 )
            v24 = 0;
          v35 = 0;
          LOBYTE(v37[0]) = 0;
          sub_2C83AE0(v24, &v35, v37);
          v10 = v35 | v33;
          v7 = (_QWORD *)v7[1];
          v33 |= v35;
          v34 |= LOBYTE(v37[0]);
        }
        while ( v7 != v31 );
        goto LABEL_11;
      }
    }
LABEL_10:
    sub_2C83CA0(v5, &v33, &v34);
    v10 = v33;
LABEL_11:
    v11 = (_QWORD *)a1[5];
    if ( !v11 )
      goto LABEL_40;
    goto LABEL_12;
  }
  return result;
}
