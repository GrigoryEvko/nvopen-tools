// Function: sub_1A95060
// Address: 0x1a95060
//
__int64 __fastcall sub_1A95060(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r14
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v13; // [rsp+28h] [rbp-68h]
  _QWORD v14[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v15[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v16; // [rsp+50h] [rbp-40h]

  v13 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v13 )
  {
    v4 = *(__int64 **)a1;
    v5 = 0;
    v6 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v7 = v5;
        v5 = *v4;
        v8 = v6;
        v6 = sub_15F4880(*v4);
        sub_15F2120(v6, a2);
        v14[0] = sub_1649960(v5);
        v16 = 773;
        v14[1] = v9;
        v15[0] = (__int64)v14;
        v15[1] = (__int64)".remat";
        sub_164B780(v6, v15);
        if ( !v8 )
          break;
        sub_1648780(v6, v7, v8);
LABEL_4:
        if ( (__int64 *)v13 == ++v4 )
          return v6;
      }
      if ( a3 == a4 )
        goto LABEL_4;
      ++v4;
      sub_1648780(v6, a3, a4);
      if ( (__int64 *)v13 == v4 )
        return v6;
    }
  }
  return 0;
}
