// Function: sub_8AEC90
// Address: 0x8aec90
//
__int64 __fastcall sub_8AEC90(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r10
  _QWORD *v8; // rdx
  char v9; // r8
  int v10; // eax
  __int64 v11; // rsi
  char v12; // cl
  __int64 v13; // rsi
  char v14; // al
  __int64 v15; // rax
  int v16; // r8d
  int v17; // eax
  char v18; // dl
  _QWORD *v19; // rdx
  __int64 v20[3]; // [rsp+8h] [rbp-18h] BYREF

  result = *(unsigned __int8 *)(a1 + 80);
  v6 = *(_QWORD *)(a1 + 88);
  if ( (_BYTE)result != 19 )
  {
    if ( (_BYTE)result == 21 )
    {
      result = *(_QWORD *)(*(_QWORD *)(v6 + 192) + 216LL);
      v7 = *(_QWORD *)result;
    }
    else
    {
      result = *(_QWORD *)(v6 + 176);
      v7 = *(_QWORD *)(result + 240);
    }
    goto LABEL_4;
  }
  v7 = *(_QWORD *)(v6 + 176);
  if ( v7 )
  {
    result = *(unsigned __int8 *)(v7 + 80);
    v19 = *(_QWORD **)(v7 + 88);
    if ( (_BYTE)result == 3 )
    {
      result = v19[21];
      v7 = *(_QWORD *)result;
    }
    else if ( (unsigned __int8)(result - 4) <= 1u )
    {
      result = v19[21];
      v7 = *(_QWORD *)(result + 168);
    }
    else if ( (_BYTE)result == 7 )
    {
      result = v19[27];
      v7 = *(_QWORD *)result;
    }
    else
    {
      v7 = v19[30];
    }
LABEL_4:
    if ( !a3 )
      goto LABEL_11;
    goto LABEL_5;
  }
  if ( a3 )
  {
LABEL_5:
    v8 = a3;
    v9 = 0;
    do
    {
      if ( (_QWORD *)a2 == v8 )
      {
        v10 = 1;
        v9 = 1;
      }
      else
      {
        v10 = v9 & 1;
      }
      v11 = v8[1];
      v12 = (_BYTE)v10 << 6;
      result = (v10 << 6) | *(_BYTE *)(v11 + 83) & 0xBFu;
      *(_BYTE *)(v11 + 83) = v12 | *(_BYTE *)(v11 + 83) & 0xBF;
      v8 = (_QWORD *)*v8;
    }
    while ( v8 );
LABEL_11:
    if ( !v7 )
      goto LABEL_16;
    v13 = *(_QWORD *)(a2 + 8);
    v14 = *(_BYTE *)(v13 + 80);
    if ( v14 == 2 )
    {
      sub_8AE7F0(a1, v13, a2, (__int64 *)v7, 1, v20);
      v15 = v20[0];
      *(_QWORD *)(a2 + 80) = v20[0];
      v16 = sub_8DC060(*(_QWORD *)(v15 + 128));
      v17 = 1;
      if ( !v16 )
      {
        v17 = v20[0];
        LOBYTE(v17) = *(_BYTE *)(v20[0] + 173) == 12;
      }
    }
    else
    {
      if ( v14 != 3 )
        sub_721090();
      result = sub_89A4B0(a1, a2, (__int64 *)v7);
      if ( !result )
      {
LABEL_16:
        if ( !*(_QWORD *)(a2 + 88) )
          goto LABEL_19;
        goto LABEL_17;
      }
      *(_QWORD *)(a2 + 80) = result;
      v17 = sub_8DC060(result) & 1;
    }
    v18 = 2 * v17;
    result = (2 * v17) | *(_BYTE *)(a2 + 56) & 0xFDu;
    *(_BYTE *)(a2 + 56) = v18 | *(_BYTE *)(a2 + 56) & 0xFD;
    goto LABEL_16;
  }
  if ( !*(_QWORD *)(a2 + 88) )
    return result;
LABEL_17:
  result = (__int64)sub_88EF90(a2);
LABEL_19:
  while ( a3 )
  {
    result = a3[1];
    *(_BYTE *)(result + 83) &= ~0x40u;
    a3 = (_QWORD *)*a3;
  }
  return result;
}
