// Function: sub_2B22CA0
// Address: 0x2b22ca0
//
__int64 *__fastcall sub_2B22CA0(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 *v5; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // rbx
  __int64 v10; // rdi
  _QWORD *v11; // rsi
  _QWORD *v12; // rax
  _BYTE *v13; // r8
  __int64 *v14; // r15
  _QWORD *v15; // rax
  _BYTE *v16; // r8
  _BYTE *v17; // r8

  v4 = (__int64)a3;
  v5 = a1;
  v7 = (a2 - (__int64)a1) >> 5;
  v8 = (a2 - (__int64)a1) >> 3;
  if ( v7 > 0 )
  {
    v9 = &a1[4 * v7];
    while ( 1 )
    {
      if ( *(_BYTE *)*v5 != 13 )
      {
        if ( *(_BYTE *)(v4 + 28) )
        {
          a4 = *(_QWORD *)(v4 + 8);
          v10 = *(unsigned int *)(v4 + 20);
          v11 = (_QWORD *)(a4 + 8 * v10);
          a3 = (_QWORD *)a4;
          if ( (_QWORD *)a4 == v11 )
            return v5;
          v12 = *(_QWORD **)(v4 + 8);
          while ( *v5 != *v12 )
          {
            if ( v11 == ++v12 )
              return v5;
          }
          v13 = (_BYTE *)v5[1];
          if ( *v13 != 13 )
          {
            v14 = v5 + 1;
            v15 = *(_QWORD **)(v4 + 8);
            goto LABEL_12;
          }
          v16 = (_BYTE *)v5[2];
          if ( *v16 != 13 )
          {
            v14 = v5 + 2;
            goto LABEL_41;
          }
          goto LABEL_25;
        }
        if ( !sub_C8CA60(v4, *v5) )
          return v5;
      }
      v13 = (_BYTE *)v5[1];
      if ( *v13 != 13 )
      {
        v14 = v5 + 1;
        if ( *(_BYTE *)(v4 + 28) )
        {
          a4 = *(_QWORD *)(v4 + 8);
          v11 = (_QWORD *)(a4 + 8LL * *(unsigned int *)(v4 + 20));
          a3 = (_QWORD *)a4;
          if ( v11 == (_QWORD *)a4 )
            return v14;
          v15 = *(_QWORD **)(v4 + 8);
LABEL_12:
          while ( v13 != (_BYTE *)*a3 )
          {
            if ( ++a3 == v11 )
              return v14;
          }
          v16 = (_BYTE *)v5[2];
          if ( *v16 != 13 )
          {
            v14 = v5 + 2;
            a3 = (_QWORD *)a4;
            goto LABEL_16;
          }
          goto LABEL_25;
        }
        if ( !sub_C8CA60(v4, v5[1]) )
          return v14;
      }
      v16 = (_BYTE *)v5[2];
      if ( *v16 != 13 )
      {
        v14 = v5 + 2;
        if ( *(_BYTE *)(v4 + 28) )
        {
          a4 = *(_QWORD *)(v4 + 8);
          v10 = *(unsigned int *)(v4 + 20);
LABEL_41:
          v11 = (_QWORD *)(a4 + 8 * v10);
          v15 = (_QWORD *)a4;
          if ( (_QWORD *)a4 == v11 )
            return v14;
          a3 = (_QWORD *)a4;
LABEL_16:
          while ( v16 != (_BYTE *)*v15 )
          {
            if ( ++v15 == v11 )
              return v14;
          }
          v17 = (_BYTE *)v5[3];
          v14 = v5 + 3;
          if ( *v17 != 13 )
            goto LABEL_28;
          goto LABEL_18;
        }
        if ( !sub_C8CA60(v4, v5[2]) )
          return v14;
      }
LABEL_25:
      v17 = (_BYTE *)v5[3];
      if ( *v17 != 13 )
      {
        v14 = v5 + 3;
        if ( *(_BYTE *)(v4 + 28) )
        {
          a3 = *(_QWORD **)(v4 + 8);
          v11 = &a3[*(unsigned int *)(v4 + 20)];
          if ( v11 == a3 )
            return v14;
LABEL_28:
          while ( v17 != (_BYTE *)*a3 )
          {
            if ( ++a3 == v11 )
              return v14;
          }
        }
        else if ( !sub_C8CA60(v4, v5[3]) )
        {
          return v14;
        }
      }
LABEL_18:
      v5 += 4;
      if ( v9 == v5 )
      {
        v8 = (a2 - (__int64)v5) >> 3;
        break;
      }
    }
  }
  if ( v8 == 2 )
  {
LABEL_49:
    if ( *(_BYTE *)*v5 != 13 && !(unsigned __int8)sub_B19060(v4, *v5, (__int64)a3, a4) )
      return v5;
    ++v5;
    goto LABEL_51;
  }
  if ( v8 == 3 )
  {
    if ( *(_BYTE *)*v5 != 13 && !(unsigned __int8)sub_B19060(v4, *v5, (__int64)a3, a4) )
      return v5;
    ++v5;
    goto LABEL_49;
  }
  if ( v8 != 1 )
    return (__int64 *)a2;
LABEL_51:
  if ( *(_BYTE *)*v5 == 13 )
    return (__int64 *)a2;
  if ( (unsigned __int8)sub_B19060(v4, *v5, (__int64)a3, a4) )
    return (__int64 *)a2;
  return v5;
}
