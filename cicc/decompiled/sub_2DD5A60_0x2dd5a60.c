// Function: sub_2DD5A60
// Address: 0x2dd5a60
//
__int64 __fastcall sub_2DD5A60(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v7; // r14
  __int64 v8; // r13
  char v9; // bl
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  __int64 v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+10h] [rbp-50h]
  __int64 v16; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+18h] [rbp-48h]

  v4 = a2 - a1;
  v5 = v4 >> 3;
  if ( v4 <= 0 )
    return a1;
  v14 = a1;
  do
  {
    while ( 1 )
    {
      v7 = v5 >> 1;
      v8 = v14 + 8 * (v5 >> 1);
      v15 = *a3;
      v17 = *(_QWORD *)(*(_QWORD *)v8 + 24LL);
      v9 = sub_AE5020(a4, v17);
      v10 = sub_9208B0(a4, v17);
      v16 = *(_QWORD *)(v15 + 24);
      v11 = ((1LL << v9) + ((unsigned __int64)(v10 + 7) >> 3) - 1) >> v9 << v9;
      LOBYTE(v17) = sub_AE5020(a4, v16);
      if ( ((1LL << v17) + ((unsigned __int64)(sub_9208B0(a4, v16) + 7) >> 3) - 1) >> v17 << v17 <= v11 )
        break;
      v14 = v8 + 8;
      v5 = v5 - v7 - 1;
      if ( v5 <= 0 )
        return v14;
    }
    v5 >>= 1;
  }
  while ( v7 > 0 );
  return v14;
}
