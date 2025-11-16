// Function: sub_2B65E20
// Address: 0x2b65e20
//
__int64 __fastcall sub_2B65E20(_QWORD *a1, __int64 a2, int a3, int a4)
{
  __int64 v5; // r12
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rsi
  __int64 v10; // r15
  int v11; // eax
  int v13; // [rsp+8h] [rbp-78h]
  char v14; // [rsp+Fh] [rbp-71h]
  __int64 v15; // [rsp+10h] [rbp-70h]
  int v16[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v17; // [rsp+28h] [rbp-58h]
  __int64 v18; // [rsp+30h] [rbp-50h]
  _QWORD *v19; // [rsp+38h] [rbp-48h]
  int v20; // [rsp+40h] [rbp-40h]
  int v21; // [rsp+44h] [rbp-3Ch]

  v5 = a3;
  v7 = a1[411];
  v8 = a1[418];
  v19 = a1;
  v9 = a1[413];
  v20 = 2;
  v18 = v7;
  *(_QWORD *)v16 = v9;
  v17 = v8;
  v21 = qword_500F9A8;
  HIDWORD(v15) = 0;
  if ( a3 )
  {
    v13 = 0;
    v10 = 0;
    v14 = 0;
    do
    {
      v11 = sub_2B65A50(
              (__int64)v16,
              *(_QWORD *)(a2 + 16LL * (int)v10),
              *(_QWORD *)(a2 + 16LL * (int)v10 + 8),
              0,
              0,
              1,
              0,
              0);
      if ( v11 > a4 )
      {
        v13 = v10;
        a4 = v11;
        v14 = 1;
      }
      ++v10;
    }
    while ( v10 != v5 );
  }
  else
  {
    v13 = 0;
    v14 = 0;
  }
  LODWORD(v15) = v13;
  BYTE4(v15) = v14;
  return v15;
}
