// Function: sub_EBC1D0
// Address: 0xebc1d0
//
__int64 __fastcall sub_EBC1D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v6; // r14
  unsigned int v7; // r13d
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // r8
  unsigned int v12; // r15d
  void (__fastcall *v13)(__int64, __int64); // rcx
  int v14; // eax
  __int64 v15; // [rsp+0h] [rbp-C0h]
  void (__fastcall *v16)(__int64, __int64); // [rsp+10h] [rbp-B0h]
  __int64 v17; // [rsp+18h] [rbp-A8h]
  __int64 v18; // [rsp+28h] [rbp-98h] BYREF
  _QWORD v19[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v20; // [rsp+50h] [rbp-70h]
  __int64 *v21; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+68h] [rbp-58h]
  const char *v23; // [rsp+70h] [rbp-50h]
  __int16 v24; // [rsp+80h] [rbp-40h]

  v6 = sub_ECD690(a1 + 40);
  if ( !*(_BYTE *)(a1 + 869) && (unsigned __int8)sub_EA2540(a1) )
    return 1;
  v7 = sub_EAC8B0(a1, &v18);
  if ( (_BYTE)v7 )
    return 1;
  if ( v18 < 0 )
  {
    v19[0] = "'";
    v24 = 770;
    v19[2] = a2;
    v20 = 1283;
    v21 = v19;
    v19[3] = a3;
    v23 = "' directive with negative repeat count has no effect";
    sub_EA8060((_QWORD *)a1, v6, (__int64)&v21, 0, 0);
    return v7;
  }
  v24 = 259;
  v21 = (__int64 *)"expected comma";
  if ( (unsigned __int8)sub_ECE210(a1, 26, &v21) )
  {
    return 1;
  }
  else
  {
    v22 = 1;
    v21 = 0;
    if ( (unsigned __int8)sub_EBB490(a1, a4, (__int64)&v21) || (v7 = sub_ECE000(a1), (_BYTE)v7) )
    {
      v7 = 1;
    }
    else
    {
      v9 = 0;
      v17 = v18;
      if ( v18 )
      {
        do
        {
          v11 = *(_QWORD *)(a1 + 232);
          v12 = v22;
          v13 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 536LL);
          if ( v22 <= 0x40 )
          {
            v10 = (__int64)v21;
          }
          else
          {
            v15 = *(_QWORD *)(a1 + 232);
            v16 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 536LL);
            v14 = sub_C444A0((__int64)&v21);
            v13 = v16;
            v10 = -1;
            v11 = v15;
            if ( v12 - v14 <= 0x40 )
              v10 = *v21;
          }
          ++v9;
          v13(v11, v10);
        }
        while ( v17 != v9 );
      }
    }
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
  }
  return v7;
}
