// Function: sub_25621E0
// Address: 0x25621e0
//
__int64 __fastcall sub_25621E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v6; // r13
  __int64 v7; // r15
  unsigned __int8 v8; // r12
  char v9; // al
  int v10; // eax
  __int64 v12; // [rsp+8h] [rbp-38h]

  v6 = a1[1];
  v7 = a1[3];
  if ( v6 )
  {
    v12 = a3;
    v8 = sub_B46420(a1[1]);
    v9 = sub_B46490(v6);
    a3 = v12;
    v10 = (2 * (v9 != 0)) | v8;
  }
  else
  {
    v10 = 3;
  }
  sub_2561E50(v7, *a1, a5, v6, a3, a1[2], v10);
  return 1;
}
