// Function: sub_1AEC470
// Address: 0x1aec470
//
__int64 __fastcall sub_1AEC470(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _QWORD *v7; // r15
  _QWORD *v8; // r14
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v13; // [rsp+0h] [rbp-40h]
  unsigned int v14; // [rsp+Ch] [rbp-34h]

  v7 = *(_QWORD **)(a1 + 8);
  v14 = 0;
  v13 = a2 + 8;
  while ( v7 )
  {
    v8 = v7;
    v7 = (_QWORD *)v7[1];
    if ( (unsigned __int8)sub_15CCFD0(a3, a4, (__int64)v8) )
    {
      if ( *v8 )
      {
        v9 = v8[1];
        v10 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v10 = v9;
        if ( v9 )
          *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
      }
      *v8 = a2;
      if ( a2 )
      {
        v11 = *(_QWORD *)(a2 + 8);
        v8[1] = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = (unsigned __int64)(v8 + 1) | *(_QWORD *)(v11 + 16) & 3LL;
        v8[2] = v13 | v8[2] & 3LL;
        *(_QWORD *)(a2 + 8) = v8;
      }
      ++v14;
    }
  }
  return v14;
}
