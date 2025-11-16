// Function: sub_2E87C90
// Address: 0x2e87c90
//
unsigned __int64 __fastcall sub_2E87C90(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  int v7; // r9d
  int v8; // eax
  unsigned __int8 *v9; // rdx
  __int64 v10; // rbx
  unsigned __int8 v11; // r10
  unsigned __int8 v12; // r9
  unsigned __int8 v13; // r11
  __int64 v14; // r11
  __int64 v15; // r12
  unsigned __int64 v16; // r9
  _QWORD *v17; // rdx

  result = *(_QWORD *)(a1 + 48);
  v5 = result & 0xFFFFFFFFFFFFFFF8LL;
  if ( (result & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    v6 = 0;
    goto LABEL_3;
  }
  v6 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v7 = result & 7;
  if ( v7 == 1 || (v6 = 0, v7 != 3) || !*(_BYTE *)(v5 + 4) )
  {
LABEL_3:
    if ( a3 == v6 )
      return result;
LABEL_4:
    if ( !a3 && (result & 7) == 1 )
    {
      *(_QWORD *)(a1 + 48) = 0;
      return result;
    }
    if ( v5 )
    {
      v8 = result & 7;
      if ( v8 == 3 )
      {
        v9 = (unsigned __int8 *)(*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL);
        v10 = 0;
        v5 = *(int *)v5;
        v11 = v9[5];
        v12 = v9[6];
        v13 = v9[7];
        if ( v9[9] )
          v10 = *(_QWORD *)&v9[8 * v13 + 16 + 8 * v12 + 8 * v5 + 8 * v11 + 8 * v9[4]];
        v8 = 0;
        if ( v9[8] )
          v8 = *(_DWORD *)&v9[8 * v5 + 16 + 8 * v11 + 8 * v9[4] + 8 * v13 + 8 * v12];
        if ( v13 )
          v14 = *(_QWORD *)&v9[8 * v12 + 16 + 8 * v5 + 8 * v11 + 8 * v9[4]];
        else
          v14 = 0;
        v15 = 0;
        if ( v12 )
          v15 = *(_QWORD *)&v9[8 * v5 + 16 + 8 * v11 + 8 * v9[4]];
        v16 = 0;
        if ( v11 )
          v16 = *(_QWORD *)&v9[8 * v5 + 16 + 8 * v9[4]];
        v17 = v9 + 16;
        return sub_2E867B0(a1, a2, v17, v5, a3, v16, v15, v14, v8, v10);
      }
      v16 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v8 != 2 )
      {
        if ( !v8 )
        {
          *(_QWORD *)(a1 + 48) = v5;
          v17 = (_QWORD *)(a1 + 48);
          v5 = 1;
          v10 = 0;
          v14 = 0;
          v15 = 0;
          v16 = 0;
          return sub_2E867B0(a1, a2, v17, v5, a3, v16, v15, v14, v8, v10);
        }
        v16 = 0;
      }
      v5 = 0;
      v17 = 0;
      v10 = 0;
      v8 = 0;
      v14 = 0;
      v15 = 0;
    }
    else
    {
      v17 = 0;
      v10 = 0;
      v8 = 0;
      v14 = 0;
      v15 = 0;
      v16 = 0;
    }
    return sub_2E867B0(a1, a2, v17, v5, a3, v16, v15, v14, v8, v10);
  }
  if ( a3 != *(_QWORD *)(v5 + 8LL * *(int *)v5 + 16) )
    goto LABEL_4;
  return result;
}
