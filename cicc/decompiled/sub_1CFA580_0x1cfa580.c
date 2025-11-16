// Function: sub_1CFA580
// Address: 0x1cfa580
//
__int64 __fastcall sub_1CFA580(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdx
  int v6; // eax
  __int64 v8; // [rsp+0h] [rbp-70h]
  __int64 v9[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v10; // [rsp+30h] [rbp-40h]

  v2 = a2 + 8;
  v3 = *(_QWORD *)(a2 + 16);
  if ( a2 + 8 != v3 )
  {
    do
    {
      while ( 1 )
      {
        v4 = v3 - 56;
        if ( !v3 )
          v4 = 0;
        sub_1649960(v4);
        if ( !v5 )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return 1;
      }
      v6 = *(_DWORD *)(a1 + 156);
      v9[0] = (__int64)"__unnamed_GV_";
      LODWORD(v8) = v6;
      *(_DWORD *)(a1 + 156) = v6 + 1;
      v9[1] = v8;
      v10 = 2307;
      sub_164B780(v4, v9);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v2 != v3 );
  }
  return 1;
}
