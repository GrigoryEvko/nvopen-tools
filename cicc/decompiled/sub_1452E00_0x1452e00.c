// Function: sub_1452E00
// Address: 0x1452e00
//
__int64 __fastcall sub_1452E00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v7; // r13
  __int64 v9; // rbx
  int v10; // r15d
  int v11; // r14d
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // r10
  __int16 v15; // r11
  __int64 v16; // rsi
  int v17; // ebx
  int v18; // r10d
  __int16 v19; // r11
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx

  result = a3 - 1;
  v5 = a2;
  v7 = a3 & 1;
  v9 = (a3 - 1) / 2;
  if ( a2 < v9 )
  {
    while ( 1 )
    {
      result = 2 * (a2 + 1);
      v12 = a1 + 16 * (a2 + 1);
      v14 = *(_QWORD *)(v12 - 8);
      v13 = *(_QWORD *)v12;
      v15 = *(_WORD *)(v14 + 24);
      if ( *(_WORD *)(*(_QWORD *)v12 + 24LL) == 5 )
      {
        v10 = *(_DWORD *)(v13 + 40);
        v11 = 1;
        if ( v15 != 5 )
          goto LABEL_5;
      }
      else
      {
        v10 = 1;
        if ( v15 != 5 )
        {
          *(_QWORD *)(a1 + 8 * a2) = v13;
          if ( result >= v9 )
            goto LABEL_12;
          goto LABEL_8;
        }
      }
      v11 = *(_DWORD *)(v14 + 40);
LABEL_5:
      if ( v11 < v10 )
      {
        --result;
        v12 = a1 + 8 * result;
        v13 = *(_QWORD *)v12;
      }
      *(_QWORD *)(a1 + 8 * a2) = v13;
      if ( result >= v9 )
      {
LABEL_12:
        if ( !v7 )
          goto LABEL_25;
LABEL_13:
        v16 = (result - 1) / 2;
        if ( result <= v5 )
          goto LABEL_28;
        while ( 2 )
        {
          v12 = a1 + 8 * v16;
          v19 = *(_WORD *)(a4 + 24);
          v20 = *(_QWORD *)v12;
          if ( *(_WORD *)(*(_QWORD *)v12 + 24LL) == 5 )
          {
            v17 = *(_DWORD *)(v20 + 40);
            v18 = 1;
            if ( v19 == 5 )
              goto LABEL_22;
          }
          else
          {
            if ( v19 != 5 )
            {
              v12 = a1 + 8 * result;
              goto LABEL_28;
            }
            v17 = 1;
LABEL_22:
            v18 = *(_DWORD *)(a4 + 40);
          }
          result = a1 + 8 * result;
          if ( v18 >= v17 )
          {
            v12 = result;
            goto LABEL_28;
          }
          *(_QWORD *)result = v20;
          result = v16;
          if ( v5 >= v16 )
            goto LABEL_28;
          v16 = (v16 - 1) / 2;
          continue;
        }
      }
LABEL_8:
      a2 = result;
    }
  }
  v12 = a1 + 8 * a2;
  if ( (a3 & 1) == 0 )
  {
    result = a2;
LABEL_25:
    if ( result == (a3 - 2) / 2 )
    {
      v21 = 2 * result + 2;
      v22 = *(_QWORD *)(a1 + 8 * v21 - 8);
      result = v21 - 1;
      *(_QWORD *)v12 = v22;
      v12 = a1 + 8 * result;
    }
    goto LABEL_13;
  }
LABEL_28:
  *(_QWORD *)v12 = a4;
  return result;
}
