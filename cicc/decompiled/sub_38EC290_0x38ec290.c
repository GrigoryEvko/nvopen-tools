// Function: sub_38EC290
// Address: 0x38ec290
//
__int64 __fastcall sub_38EC290(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r14
  unsigned __int64 v5; // r15
  unsigned int v6; // r13d
  __int64 v7; // r15
  __int64 v8; // rbx
  _QWORD v10[3]; // [rsp+0h] [rbp-90h] BYREF
  __int64 v11; // [rsp+18h] [rbp-78h] BYREF
  char *v12; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v13; // [rsp+28h] [rbp-68h]
  __int16 v14; // [rsp+30h] [rbp-60h]
  char **v15; // [rsp+40h] [rbp-50h] BYREF
  char *v16; // [rsp+48h] [rbp-48h]
  __int16 v17; // [rsp+50h] [rbp-40h]

  v4 = a4;
  v10[0] = a2;
  v10[1] = a3;
  v5 = sub_3909290(a1 + 144);
  if ( !*(_BYTE *)(a1 + 845) && (unsigned __int8)sub_38E36C0(a1) )
    return 1;
  v6 = sub_38EB9C0(a1, &v11);
  if ( (_BYTE)v6 )
    return 1;
  if ( v11 < 0 )
  {
    v14 = 1283;
    v12 = "'";
    v13 = v10;
    v15 = &v12;
    v17 = 770;
    v16 = "' directive with negative repeat count has no effect";
    sub_38E4170((_QWORD *)a1, v5, (__int64)&v15, 0, 0);
    return v6;
  }
  v12 = "unexpected token in '";
  v13 = v10;
  v14 = 1283;
  v15 = &v12;
  v17 = 770;
  v16 = "' directive";
  v6 = sub_3909E20(a1, 9, &v15);
  if ( (_BYTE)v6 )
  {
    return 1;
  }
  else
  {
    v7 = v11;
    v8 = 0;
    if ( v11 )
    {
      do
      {
        ++v8;
        sub_38DD0A0(*(__int64 **)(a1 + 328), v4, 0);
      }
      while ( v7 != v8 );
    }
  }
  return v6;
}
