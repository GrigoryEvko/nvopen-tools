// Function: sub_3175FF0
// Address: 0x3175ff0
//
unsigned __int8 *__fastcall sub_3175FF0(__int64 a1, unsigned __int8 *a2)
{
  __int64 *v3; // r12
  __int64 *v4; // rdx
  __int64 *v5; // rsi
  unsigned __int64 v6; // rax
  int v7; // edi
  unsigned __int8 *result; // rax
  __int64 *v9; // r12
  __m128i v10; // [rsp+0h] [rbp-70h] BYREF
  __int64 v11; // [rsp+10h] [rbp-60h]
  __int64 v12; // [rsp+18h] [rbp-58h]
  __int64 v13; // [rsp+20h] [rbp-50h]
  __int64 v14; // [rsp+28h] [rbp-48h]
  __int64 v15; // [rsp+30h] [rbp-40h]
  __int64 v16; // [rsp+38h] [rbp-38h]
  __int16 v17; // [rsp+40h] [rbp-30h]

  v3 = (__int64 *)*((_QWORD *)a2 - 4);
  if ( **(__int64 ***)(a1 + 240) == v3 )
  {
    v9 = (__int64 *)*((_QWORD *)a2 - 8);
    v5 = (__int64 *)sub_31751A0(a1, v9);
    if ( !v5 )
      v5 = v9;
    v4 = *(__int64 **)(*(_QWORD *)(a1 + 240) + 8LL);
  }
  else
  {
    v4 = (__int64 *)sub_31751A0(a1, *((_BYTE **)a2 - 4));
    v5 = *(__int64 **)(*(_QWORD *)(a1 + 240) + 8LL);
    if ( !v4 )
      v4 = v3;
  }
  v6 = *(_QWORD *)(a1 + 40);
  v7 = *a2;
  v11 = 0;
  v10 = (__m128i)v6;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 257;
  result = sub_101E7C0(v7 - 29, v5, v4, &v10);
  if ( result )
  {
    if ( *result >= 0x16u )
      return 0;
  }
  return result;
}
