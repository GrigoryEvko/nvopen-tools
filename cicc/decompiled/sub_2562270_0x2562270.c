// Function: sub_2562270
// Address: 0x2562270
//
__int64 __fastcall sub_2562270(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v7; // r14
  __int64 v8; // r9
  unsigned __int8 v9; // bl
  char v10; // al
  int v11; // eax
  __int64 v13; // [rsp+8h] [rbp-38h]

  v7 = *a1;
  v8 = a1[1];
  if ( a2 )
  {
    v13 = a1[1];
    v9 = sub_B46420(a2);
    v10 = sub_B46490(a2);
    v8 = v13;
    v11 = (2 * (v10 != 0)) | v9;
  }
  else
  {
    v11 = 3;
  }
  sub_2561E50(v7, v7 + 88, a5, a2, a3, v8, v11);
  return 1;
}
