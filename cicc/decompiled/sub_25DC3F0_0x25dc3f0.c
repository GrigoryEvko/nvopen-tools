// Function: sub_25DC3F0
// Address: 0x25dc3f0
//
__int64 __fastcall sub_25DC3F0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r8
  __int64 v7; // r10
  __int64 v8; // rbx
  __int64 result; // rax
  _QWORD *v10; // rdx
  unsigned __int64 v11; // r9
  _QWORD *v12; // rcx
  __int64 v13; // rbx
  __int64 v14; // r11
  unsigned __int64 v15; // r10
  __int64 v16; // rcx
  _QWORD *v17; // rsi
  unsigned __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rcx
  _QWORD *v22; // rcx

  v4 = a2;
  v7 = (a3 - 1) / 2;
  v8 = a3 & 1;
  if ( a2 >= v7 )
  {
    result = 3 * a2;
    v10 = (_QWORD *)(a1 + 24 * a2);
    if ( v8 )
    {
      v13 = *a4;
      v14 = a4[1];
      v15 = a4[2];
      goto LABEL_13;
    }
    result = a2;
    goto LABEL_16;
  }
  while ( 1 )
  {
    result = 2 * (a2 + 1);
    v10 = (_QWORD *)(a1 + 48 * (a2 + 1));
    v11 = v10[2];
    if ( v11 < *(v10 - 1) )
    {
      --result;
      v10 = (_QWORD *)(a1 + 24 * result);
      v11 = v10[2];
    }
    v12 = (_QWORD *)(a1 + 24 * a2);
    v12[2] = v11;
    v12[1] = v10[1];
    *v12 = *v10;
    if ( result >= v7 )
      break;
    a2 = result;
  }
  if ( !v8 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == result )
    {
      v19 = result + 1;
      v20 = 2 * (result + 1);
      v21 = v20 + 4 * v19;
      result = v20 - 1;
      v22 = (_QWORD *)(a1 + 8 * v21 - 24);
      v10[2] = v22[2];
      v10[1] = v22[1];
      *v10 = *v22;
      v10 = (_QWORD *)(a1 + 24 * result);
    }
  }
  v13 = *a4;
  v14 = a4[1];
  v15 = a4[2];
  v16 = (result - 1) / 2;
  if ( result > v4 )
  {
    while ( 1 )
    {
      result *= 3;
      v17 = (_QWORD *)(a1 + 24 * v16);
      v10 = (_QWORD *)(a1 + 8 * result);
      v18 = v17[2];
      if ( v15 <= v18 )
        break;
      v10[2] = v18;
      v10[1] = v17[1];
      *v10 = *v17;
      result = v16;
      if ( v4 >= v16 )
      {
        v17[2] = v15;
        v17[1] = v14;
        *v17 = v13;
        return result;
      }
      v16 = (v16 - 1) / 2;
    }
  }
LABEL_13:
  v10[2] = v15;
  v10[1] = v14;
  *v10 = v13;
  return result;
}
