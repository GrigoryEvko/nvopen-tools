// Function: sub_38C6700
// Address: 0x38c6700
//
_BYTE *__fastcall sub_38C6700(__int64 a1, int a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rsi
  _BYTE *v12; // rcx
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // r14
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rbx
  char v20; // si
  char v21; // al
  char *v22; // rax
  _BYTE *result; // rax
  unsigned __int64 v24; // rdx
  char v25; // r14
  char v26; // si
  char *v27; // rax
  char v28; // al
  _BYTE *v29; // rax
  char v30; // si
  char v31; // al
  char *v32; // rax
  __int64 v33; // [rsp+10h] [rbp-50h]
  unsigned __int64 v34; // [rsp+18h] [rbp-48h]
  char v35; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+20h] [rbp-40h]
  char v38; // [rsp+29h] [rbp-37h]

  v5 = a3;
  v9 = BYTE2(a2);
  v33 = (unsigned __int8)a2;
  v38 = BYTE1(a2);
  v34 = (255 - (unsigned __int64)(unsigned __int8)a2) / BYTE2(a2);
  v10 = *(unsigned int *)(*(_QWORD *)(a1 + 16) + 28LL);
  v12 = *(_BYTE **)(a5 + 24);
  v13 = a4 / v10;
  v14 = a4 / v10;
  v15 = *(_QWORD *)(a5 + 16);
  if ( a3 == 0x7FFFFFFFFFFFFFFFLL )
  {
    if ( v13 == v34 )
    {
      if ( v15 <= (unsigned __int64)v12 )
      {
        sub_16E7DE0(a5, 8);
        v12 = *(_BYTE **)(a5 + 24);
      }
      else
      {
        *(_QWORD *)(a5 + 24) = v12 + 1;
        *v12 = 8;
        v12 = *(_BYTE **)(a5 + 24);
      }
    }
    else if ( a4 >= v10 )
    {
      if ( v15 <= (unsigned __int64)v12 )
      {
        sub_16E7DE0(a5, 2);
      }
      else
      {
        *(_QWORD *)(a5 + 24) = v12 + 1;
        *v12 = 2;
      }
      do
      {
        v30 = v14 & 0x7F;
        v31 = v14 & 0x7F | 0x80;
        v14 >>= 7;
        if ( v14 )
          v30 = v31;
        v32 = *(char **)(a5 + 24);
        if ( (unsigned __int64)v32 < *(_QWORD *)(a5 + 16) )
        {
          *(_QWORD *)(a5 + 24) = v32 + 1;
          *v32 = v30;
        }
        else
        {
          sub_16E7DE0(a5, v30);
        }
      }
      while ( v14 );
      v12 = *(_BYTE **)(a5 + 24);
    }
    if ( (unsigned __int64)v12 >= *(_QWORD *)(a5 + 16) )
    {
      sub_16E7DE0(a5, 0);
    }
    else
    {
      *(_QWORD *)(a5 + 24) = v12 + 1;
      *v12 = 0;
    }
    v29 = *(_BYTE **)(a5 + 24);
    if ( (unsigned __int64)v29 >= *(_QWORD *)(a5 + 16) )
    {
      sub_16E7DE0(a5, 1);
    }
    else
    {
      *(_QWORD *)(a5 + 24) = v29 + 1;
      *v29 = 1;
    }
    result = *(_BYTE **)(a5 + 24);
    if ( (unsigned __int64)result >= *(_QWORD *)(a5 + 16) )
      goto LABEL_39;
    goto LABEL_17;
  }
  v16 = a3 - v38;
  if ( v16 < v9 )
  {
    v35 = 0;
    if ( v16 + v33 <= 0xFF )
    {
      if ( !(v13 | a3) )
        goto LABEL_30;
LABEL_5:
      v17 = v16 + v33;
      if ( v34 + 256 > v13 )
      {
        v18 = v17 + v9 * v13;
        if ( v18 <= 0xFF )
        {
          if ( (unsigned __int64)v12 >= v15 )
            return (_BYTE *)sub_16E7DE0(a5, v18);
          *(_QWORD *)(a5 + 24) = v12 + 1;
          *v12 = v18;
          return v12 + 1;
        }
        v19 = v17 + (v13 - v34) * v9;
        if ( v19 <= 0xFF )
        {
          if ( (unsigned __int64)v12 >= v15 )
          {
            sub_16E7DE0(a5, 8);
          }
          else
          {
            *(_QWORD *)(a5 + 24) = v12 + 1;
            *v12 = 8;
          }
          result = *(_BYTE **)(a5 + 24);
          LOBYTE(v18) = v19;
          if ( (unsigned __int64)result >= *(_QWORD *)(a5 + 16) )
            return (_BYTE *)sub_16E7DE0(a5, v18);
          *(_QWORD *)(a5 + 24) = result + 1;
          *result = v19;
          return result;
        }
      }
      if ( (unsigned __int64)v12 >= v15 )
      {
        sub_16E7DE0(a5, 2);
      }
      else
      {
        *(_QWORD *)(a5 + 24) = v12 + 1;
        *v12 = 2;
      }
      do
      {
        while ( 1 )
        {
          v20 = v14 & 0x7F;
          v21 = v14 & 0x7F | 0x80;
          v14 >>= 7;
          if ( v14 )
            v20 = v21;
          v22 = *(char **)(a5 + 24);
          if ( (unsigned __int64)v22 >= *(_QWORD *)(a5 + 16) )
            break;
          *(_QWORD *)(a5 + 24) = v22 + 1;
          *v22 = v20;
          if ( !v14 )
            goto LABEL_15;
        }
        sub_16E7DE0(a5, v20);
      }
      while ( v14 );
LABEL_15:
      result = *(_BYTE **)(a5 + 24);
      v24 = *(_QWORD *)(a5 + 16);
      if ( !v35 )
      {
        if ( v24 <= (unsigned __int64)result )
        {
          LOBYTE(v18) = v17;
          return (_BYTE *)sub_16E7DE0(a5, v18);
        }
        *(_QWORD *)(a5 + 24) = result + 1;
        *result = v17;
        return result;
      }
      if ( v24 <= (unsigned __int64)result )
        goto LABEL_39;
LABEL_17:
      *(_QWORD *)(a5 + 24) = result + 1;
      *result = 1;
      return result;
    }
  }
  if ( v15 <= (unsigned __int64)v12 )
  {
    sub_16E7DE0(a5, 3);
    v5 = a3;
  }
  else
  {
    *(_QWORD *)(a5 + 24) = v12 + 1;
    *v12 = 3;
  }
  do
  {
    while ( 1 )
    {
      v28 = v5;
      v26 = v5 & 0x7F;
      v5 >>= 7;
      if ( !v5 )
      {
        v25 = 0;
        if ( (v28 & 0x40) == 0 )
          goto LABEL_22;
        goto LABEL_21;
      }
      if ( v5 == -1 )
      {
        v25 = 0;
        if ( (v28 & 0x40) != 0 )
          break;
      }
LABEL_21:
      v26 |= 0x80u;
      v25 = 1;
LABEL_22:
      v27 = *(char **)(a5 + 24);
      if ( (unsigned __int64)v27 >= *(_QWORD *)(a5 + 16) )
        goto LABEL_28;
LABEL_23:
      *(_QWORD *)(a5 + 24) = v27 + 1;
      *v27 = v26;
      if ( !v25 )
        goto LABEL_29;
    }
    v27 = *(char **)(a5 + 24);
    if ( (unsigned __int64)v27 < *(_QWORD *)(a5 + 16) )
      goto LABEL_23;
LABEL_28:
    v36 = v5;
    sub_16E7DE0(a5, v26);
    v5 = v36;
  }
  while ( v25 );
LABEL_29:
  v12 = *(_BYTE **)(a5 + 24);
  v35 = 1;
  v16 = -v38;
  v15 = *(_QWORD *)(a5 + 16);
  if ( v13 )
    goto LABEL_5;
LABEL_30:
  if ( (unsigned __int64)v12 < v15 )
  {
    *(_QWORD *)(a5 + 24) = v12 + 1;
    *v12 = 1;
    return v12 + 1;
  }
LABEL_39:
  LOBYTE(v18) = 1;
  return (_BYTE *)sub_16E7DE0(a5, v18);
}
