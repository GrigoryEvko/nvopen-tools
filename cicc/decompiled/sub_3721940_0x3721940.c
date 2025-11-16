// Function: sub_3721940
// Address: 0x3721940
//
__int64 __fastcall sub_3721940(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  unsigned __int8 *v4; // r12
  __int64 v5; // rdx
  int v6; // eax
  const char *v8; // [rsp+0h] [rbp-60h] BYREF
  int v9; // [rsp+10h] [rbp-50h]
  __int16 v10; // [rsp+20h] [rbp-40h]

  v2 = a2 + 8;
  v3 = *(_QWORD *)(a2 + 16);
  if ( v3 != a2 + 8 )
  {
    do
    {
      while ( 1 )
      {
        v4 = (unsigned __int8 *)(v3 - 56);
        if ( !v3 )
          v4 = 0;
        sub_BD5D20((__int64)v4);
        if ( !v5 )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return 1;
      }
      v6 = *(_DWORD *)(a1 + 172);
      v8 = "__unnamed_GV_";
      v9 = v6;
      *(_DWORD *)(a1 + 172) = v6 + 1;
      v10 = 2307;
      sub_BD6B50(v4, &v8);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v2 != v3 );
  }
  return 1;
}
