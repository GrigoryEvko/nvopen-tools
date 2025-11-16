// Function: sub_1BB44E0
// Address: 0x1bb44e0
//
unsigned __int64 __fastcall sub_1BB44E0(__int64 a1, char a2, unsigned int a3, __m128i a4, __m128i a5)
{
  unsigned int v7; // r14d
  unsigned int v8; // ebx
  int v9; // eax
  __int64 v10; // rsi
  unsigned int v11; // edx
  int v12; // ecx
  __int64 v13; // r13
  int v14; // eax
  __int64 v16; // r13
  int v17; // edi
  unsigned int v18; // [rsp+Ch] [rbp-34h] BYREF
  __int64 v19; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-28h] BYREF
  char v21; // [rsp+1Ch] [rbp-24h]

  sub_1BB4260((__int64)&v20, *(__int64 **)(a1 + 40), a2, a4, a5);
  if ( !v21 )
    return 1;
  if ( a3 )
  {
    v18 = a3;
    v16 = *(_QWORD *)(a1 + 40);
    if ( a3 != 1 && !(unsigned __int8)sub_1B97860(v16 + 168, (int *)&v18, &v19) )
    {
      sub_1BA5EB0(v16, v18, a4, a5);
      sub_1BB12A0(v16, v18);
      sub_1BAFFE0(v16, v18);
    }
    sub_1BA8A90(v16, a3);
    sub_1BACC80(a1, a3, a3);
    return a3;
  }
  v7 = v20;
  if ( !v20 )
  {
    sub_1BACC80(a1, 1u, 0);
    return sub_1BA88C0(*(_QWORD *)(a1 + 40), v7, *(double *)a4.m128i_i64);
  }
  v8 = 1;
  do
  {
    if ( v8 == 1 )
      goto LABEL_8;
    v13 = *(_QWORD *)(a1 + 40);
    v14 = *(_DWORD *)(v13 + 192);
    if ( v14 )
    {
      v9 = v14 - 1;
      v10 = *(_QWORD *)(v13 + 176);
      v11 = v9 & (37 * v8);
      v12 = *(_DWORD *)(v10 + 80LL * v11);
      if ( v8 == v12 )
        goto LABEL_6;
      v17 = 1;
      while ( v12 != -1 )
      {
        v11 = v9 & (v17 + v11);
        v12 = *(_DWORD *)(v10 + 80LL * v11);
        if ( v8 == v12 )
          goto LABEL_6;
        ++v17;
      }
    }
    sub_1BA5EB0(*(_QWORD *)(a1 + 40), v8, a4, a5);
    sub_1BB12A0(v13, v8);
    sub_1BAFFE0(v13, v8);
LABEL_6:
    if ( v8 > 1 )
      sub_1BA8A90(*(_QWORD *)(a1 + 40), v8);
LABEL_8:
    v8 *= 2;
  }
  while ( v7 >= v8 );
  sub_1BACC80(a1, 1u, v7);
  if ( v7 == 1 )
    return 1;
  return sub_1BA88C0(*(_QWORD *)(a1 + 40), v7, *(double *)a4.m128i_i64);
}
