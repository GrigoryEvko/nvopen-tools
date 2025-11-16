// Function: sub_30887C0
// Address: 0x30887c0
//
__int64 __fastcall sub_30887C0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  char v8; // r12
  __int64 j; // r14
  unsigned __int8 *v10; // rsi
  __int64 i; // [rsp+8h] [rbp-68h]
  __int64 v13; // [rsp+18h] [rbp-58h]
  __int64 v14; // [rsp+20h] [rbp-50h]
  __int64 v15; // [rsp+28h] [rbp-48h]
  unsigned __int8 *v16; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int8 *v17; // [rsp+38h] [rbp-38h] BYREF

  v6 = a1 + 176;
  v7 = sub_30885F0((_QWORD *)(a1 + 176), a2);
  v13 = sub_30885F0((_QWORD *)(a1 + 176), *(_QWORD *)(a3 - 32));
  v15 = v13 + 8;
  v14 = *(_QWORD *)(v7 + 24);
  for ( i = v7 + 8; i != v14; v14 = sub_220EF30(v14) )
  {
    v16 = 0;
    v8 = sub_30857E0(v6, *(unsigned __int8 **)(v14 + 32), &v16, a4);
    for ( j = *(_QWORD *)(v13 + 24); v15 != j; j = sub_220EF30(j) )
    {
      v10 = *(unsigned __int8 **)(j + 32);
      v17 = 0;
      if ( (unsigned __int8)sub_30857E0(v6, v10, &v17, a4) )
      {
        if ( v8 && (!v16 || !v17 || v16 == v17) )
          return 1;
      }
      else if ( !v8 )
      {
        return 1;
      }
    }
  }
  return 0;
}
