// Function: sub_822DE0
// Address: 0x822de0
//
_QWORD *__fastcall sub_822DE0(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *result; // rax
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // r8
  __int64 v9; // rbx
  int v10; // r11d
  __int64 *v11; // r12
  _QWORD *v12; // r10
  __int64 *v13; // rdi
  _QWORD *v14; // rdx
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  char *v17; // rdx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdx

  result = (_QWORD *)qword_4F195E0;
  v7 = a2 + 48;
  v8 = (_DWORD)a4 == 0 ? 0x10000LL : 2048LL;
  v9 = a1;
  if ( qword_4F195E0 )
  {
    a6 = a3;
    v10 = a4;
    v11 = 0;
    v12 = 0;
    v13 = 0;
    while ( 1 )
    {
      a4 = result[1];
      v15 = result[3] - a4 + 48;
      if ( v15 < v7 )
        goto LABEL_4;
      if ( a4 == a6 )
        goto LABEL_8;
      if ( v12 || v10 && v8 < v15 )
      {
LABEL_4:
        v14 = (_QWORD *)*result;
        v13 = result;
        if ( !*result )
          goto LABEL_15;
      }
      else
      {
        if ( !a6 )
        {
LABEL_8:
          v16 = *result;
          if ( !v13 )
            goto LABEL_25;
          goto LABEL_9;
        }
        v14 = (_QWORD *)*result;
        v11 = v13;
        v12 = result;
        v13 = result;
        if ( !*result )
        {
LABEL_15:
          if ( !v12 )
            break;
          result = v12;
          v13 = v11;
          v16 = *v12;
          if ( !v11 )
          {
LABEL_25:
            qword_4F195E0 = v16;
            v17 = (char *)result[1];
            goto LABEL_10;
          }
LABEL_9:
          *v13 = v16;
          v17 = (char *)result[1];
          goto LABEL_10;
        }
      }
      result = v14;
    }
  }
  if ( v8 >= v7 )
    v7 = v8;
  v18 = v7;
  v19 = v7 & 7;
  if ( (v7 & 7) != 0 )
    v18 = 8 - (int)v19 + v7;
  if ( unk_4D04508 )
  {
    v18 = sub_721A20(v18);
    result = sub_822CD0(v18);
  }
  else
  {
    result = (_QWORD *)malloc(v18, v7, v19, a4, v8, a6);
    if ( !result )
      sub_685240(4u);
  }
  v17 = (char *)(result + 6);
  result[4] = v18;
  result[1] = result + 6;
  result[3] = (char *)result + v18;
LABEL_10:
  result[2] = v17;
  *((_BYTE *)result + 40) = 0;
  *result = *(_QWORD *)(unk_4F073B0 + 8 * v9);
  *(_QWORD *)(unk_4F073B0 + 8 * v9) = result;
  return result;
}
