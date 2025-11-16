// Function: sub_ADD7A0
// Address: 0xadd7a0
//
unsigned __int64 __fastcall sub_ADD7A0(__int64 a1, __int64 a2, __int64 a3, __int16 a4, char a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r15
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  int v11; // ecx
  _QWORD *v12; // rdx
  __int64 v13; // r14
  unsigned __int64 v14; // rsi
  _QWORD *v15; // rax
  int v16; // ecx
  _QWORD *v17; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rsi
  _QWORD v22[7]; // [rsp+18h] [rbp-38h] BYREF

  HIBYTE(a4) = a5;
  v6 = *(_QWORD *)(a3 + 16);
  *(_QWORD *)(a1 + 48) = v6;
  *(_QWORD *)(a1 + 56) = a3;
  *(_WORD *)(a1 + 64) = a4;
  if ( a3 != v6 + 48 )
  {
    v7 = *(_QWORD *)sub_B46C60(a3 - 24);
    v22[0] = v7;
    if ( v7 && (sub_B96E90(v22, v7, 1), (v8 = v22[0]) != 0) )
    {
      v9 = *(unsigned int *)(a1 + 8);
      v10 = *(_QWORD **)a1;
      v11 = *(_DWORD *)(a1 + 8);
      v12 = (_QWORD *)(*(_QWORD *)a1 + 16 * v9);
      if ( *(_QWORD **)a1 != v12 )
      {
        while ( *(_DWORD *)v10 )
        {
          v10 += 2;
          if ( v12 == v10 )
            goto LABEL_26;
        }
        v10[1] = v22[0];
        goto LABEL_9;
      }
LABEL_26:
      v19 = *(unsigned int *)(a1 + 12);
      if ( v9 >= v19 )
      {
        v21 = v9 + 1;
        if ( v19 < v21 )
        {
          sub_C8D5F0(a1, a1 + 16, v21, 16);
          v12 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
        }
        *v12 = 0;
        v12[1] = v8;
        v8 = v22[0];
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        if ( v12 )
        {
          *(_DWORD *)v12 = 0;
          v12[1] = v8;
          v11 = *(_DWORD *)(a1 + 8);
          v8 = v22[0];
        }
        *(_DWORD *)(a1 + 8) = v11 + 1;
      }
    }
    else
    {
      sub_93FB40(a1, 0);
      v8 = v22[0];
    }
    if ( v8 )
LABEL_9:
      sub_B91220(v22);
  }
  sub_B10CB0(v22, a2);
  v13 = v22[0];
  if ( v22[0] )
  {
    v14 = *(unsigned int *)(a1 + 8);
    v15 = *(_QWORD **)a1;
    v16 = *(_DWORD *)(a1 + 8);
    v17 = (_QWORD *)(*(_QWORD *)a1 + 16 * v14);
    if ( *(_QWORD **)a1 != v17 )
    {
      while ( *(_DWORD *)v15 )
      {
        v15 += 2;
        if ( v17 == v15 )
          goto LABEL_20;
      }
      v15[1] = v22[0];
      return sub_B91220(v22);
    }
LABEL_20:
    result = *(unsigned int *)(a1 + 12);
    if ( v14 >= result )
    {
      v20 = v14 + 1;
      if ( result < v20 )
      {
        result = sub_C8D5F0(a1, a1 + 16, v20, 16);
        v17 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
      }
      *v17 = 0;
      v17[1] = v13;
      v13 = v22[0];
      ++*(_DWORD *)(a1 + 8);
    }
    else
    {
      if ( v17 )
      {
        *(_DWORD *)v17 = 0;
        v17[1] = v13;
        v16 = *(_DWORD *)(a1 + 8);
        v13 = v22[0];
      }
      *(_DWORD *)(a1 + 8) = v16 + 1;
    }
  }
  else
  {
    result = (unsigned __int64)sub_93FB40(a1, 0);
    v13 = v22[0];
  }
  if ( v13 )
    return sub_B91220(v22);
  return result;
}
