// Function: sub_2DD5BA0
// Address: 0x2dd5ba0
//
_QWORD *__fastcall sub_2DD5BA0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v8; // r13
  char v9; // bl
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  __int64 *v13; // [rsp+0h] [rbp-60h]
  _QWORD *v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+10h] [rbp-50h]
  __int64 v16; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+18h] [rbp-48h]

  v4 = a2 - (_QWORD)a1;
  v5 = v4 >> 3;
  if ( v4 <= 0 )
    return a1;
  v14 = a1;
  do
  {
    while ( 1 )
    {
      v8 = v5 >> 1;
      v13 = &v14[v5 >> 1];
      v15 = *v13;
      v17 = *(_QWORD *)(*(_QWORD *)a3 + 24LL);
      v9 = sub_AE5020(a4, v17);
      v10 = sub_9208B0(a4, v17);
      v16 = *(_QWORD *)(v15 + 24);
      v11 = ((1LL << v9) + ((unsigned __int64)(v10 + 7) >> 3) - 1) >> v9 << v9;
      LOBYTE(v17) = sub_AE5020(a4, v16);
      if ( ((1LL << v17) + ((unsigned __int64)(sub_9208B0(a4, v16) + 7) >> 3) - 1) >> v17 << v17 > v11 )
        break;
      v5 = v5 - v8 - 1;
      v14 = v13 + 1;
      if ( v5 <= 0 )
        return v14;
    }
    v5 >>= 1;
  }
  while ( v8 > 0 );
  return v14;
}
