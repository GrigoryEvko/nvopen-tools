// Function: sub_810D50
// Address: 0x810d50
//
unsigned __int8 *__fastcall sub_810D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  _QWORD v9[4]; // [rsp+0h] [rbp-70h] BYREF
  char v10; // [rsp+20h] [rbp-50h]
  __int64 v11; // [rsp+28h] [rbp-48h]
  __int64 v12; // [rsp+30h] [rbp-40h]
  int v13; // [rsp+38h] [rbp-38h]
  char v14; // [rsp+3Ch] [rbp-34h]
  __int64 v15; // [rsp+40h] [rbp-30h]

  v6 = a1;
  v9[3] = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  sub_809110(a1, a2, a3, a4, a5, a6, 0, 0, 0);
  sub_823800(qword_4F18BE0);
  sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
  v9[0] += 4LL;
  sub_8238B0(qword_4F18BE0, "TV", 2);
  if ( a2 )
  {
    sub_810BA0(a2, v9);
    v9[0] += 2LL;
    sub_8238B0(qword_4F18BE0, &unk_42B6DB2, 2);
  }
  if ( a3 )
  {
    sub_810BA0(a3, v9);
    v9[0] += 2LL;
    sub_8238B0(qword_4F18BE0, &unk_42B6DB2, 2);
    v6 = *(_QWORD *)(a3 + 56);
  }
  sub_810650(v6, 1, v9);
  return sub_80B290(0, 1, (__int64)v9);
}
