// Function: sub_2E86A90
// Address: 0x2e86a90
//
unsigned __int64 __fastcall sub_2E86A90(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // rax
  _BYTE *v5; // r8
  int v6; // eax
  __int64 v7; // r12
  unsigned __int8 v8; // r10
  unsigned __int8 v9; // r9
  unsigned __int8 v10; // r11
  unsigned __int8 v11; // bl
  int v12; // r13d
  __int64 v13; // rbx
  __int64 v14; // r11
  unsigned __int64 v15; // r9

  if ( a4 )
  {
    v4 = *(_QWORD *)(a1 + 48);
    v5 = (_BYTE *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v6 = v4 & 7;
      if ( v6 != 3 )
      {
        if ( v6 == 2 )
        {
          v15 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          v7 = 0;
          v12 = 0;
          v13 = 0;
          v14 = 0;
          v5 = 0;
        }
        else
        {
          v7 = 0;
          v12 = 0;
          v13 = 0;
          v14 = 0;
          v15 = 0;
          if ( v6 != 1 )
            v5 = 0;
        }
        return sub_2E867B0(a1, a2, a3, a4, (__int64)v5, v15, v14, v13, v12, v7);
      }
      v7 = 0;
      v8 = v5[4];
      v9 = v5[5];
      v10 = v5[6];
      v11 = v5[7];
      if ( v5[9] )
        v7 = *(_QWORD *)&v5[8 * v11 + 16 + 8 * v10 + 8 * v9 + 8 * v8 + 8 * (__int64)*(int *)v5];
      v12 = 0;
      if ( v5[8] )
        v12 = *(_DWORD *)&v5[8 * *(int *)v5 + 16 + 8 * v11 + 8 * v10 + 8 * (__int64)(v9 + v8)];
      if ( v11 )
        v13 = *(_QWORD *)&v5[8 * v10 + 16 + 8 * *(int *)v5 + 8 * (__int64)(v9 + v8)];
      else
        v13 = 0;
      if ( v10 )
        v14 = *(_QWORD *)&v5[8 * *(int *)v5 + 16 + 8 * (__int64)(v9 + v8)];
      else
        v14 = 0;
      if ( v9 )
        v15 = *(_QWORD *)&v5[8 * v8 + 16 + 8 * (__int64)*(int *)v5];
      else
        v15 = 0;
      if ( v8 )
      {
        v5 = *(_BYTE **)&v5[8 * *(int *)v5 + 16];
        return sub_2E867B0(a1, a2, a3, a4, (__int64)v5, v15, v14, v13, v12, v7);
      }
    }
    else
    {
      v7 = 0;
      v12 = 0;
      v13 = 0;
      v14 = 0;
      v15 = 0;
    }
    v5 = 0;
    return sub_2E867B0(a1, a2, a3, a4, (__int64)v5, v15, v14, v13, v12, v7);
  }
  return sub_2E868D0(a1, a2);
}
