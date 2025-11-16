// Function: sub_EAD790
// Address: 0xead790
//
__int64 __fastcall sub_EAD790(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // rbx
  unsigned __int64 v5; // r13
  unsigned int v6; // r12d
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v11; // [rsp+18h] [rbp-98h] BYREF
  _QWORD v12[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v13; // [rsp+40h] [rbp-70h]
  _QWORD v14[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v15; // [rsp+70h] [rbp-40h]

  v4 = a4;
  v5 = sub_ECD690(a1 + 40);
  if ( !*(_BYTE *)(a1 + 869) && (unsigned __int8)sub_EA2540(a1) )
    return 1;
  if ( (unsigned __int8)sub_EAC8B0(a1, &v11) )
    return 1;
  v6 = sub_ECE000(a1);
  if ( (_BYTE)v6 )
  {
    return 1;
  }
  else
  {
    v8 = v11;
    if ( v11 < 0 )
    {
      v12[0] = "'";
      v15 = 770;
      v12[2] = a2;
      v12[3] = a3;
      v13 = 1283;
      v14[0] = v12;
      v14[2] = "' directive with negative repeat count has no effect";
      sub_EA8060((_QWORD *)a1, v5, (__int64)v14, 0, 0);
    }
    else
    {
      v9 = 0;
      if ( v11 )
      {
        do
        {
          ++v9;
          sub_E99280(*(_QWORD ***)(a1 + 232), v4, 0);
        }
        while ( v8 != v9 );
      }
    }
  }
  return v6;
}
