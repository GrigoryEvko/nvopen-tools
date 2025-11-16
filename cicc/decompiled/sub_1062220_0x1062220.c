// Function: sub_1062220
// Address: 0x1062220
//
__int64 __fastcall sub_1062220(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r13d
  unsigned int v5; // r8d
  __int64 v7; // r15
  int v8; // r13d
  __int64 *v9; // r14
  unsigned int v10; // eax
  bool v11; // al
  int v12; // r14d
  __int64 v13; // [rsp+10h] [rbp-50h]
  __int64 v14; // [rsp+18h] [rbp-48h]
  __int64 *v15; // [rsp+20h] [rbp-40h]
  int v16; // [rsp+28h] [rbp-38h]
  unsigned int i; // [rsp+2Ch] [rbp-34h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    v8 = v4 - 1;
    v14 = sub_1061AC0();
    v13 = sub_1061AD0();
    v16 = 1;
    v15 = 0;
    for ( i = v8 & sub_1061E50(*a2); ; i = v12 )
    {
      v9 = (__int64 *)(v7 + 8LL * i);
      LOBYTE(v10) = sub_1061B40(*a2, *v9);
      v5 = v10;
      if ( (_BYTE)v10 )
      {
        *a3 = v9;
        return v5;
      }
      if ( sub_1061B40(*v9, v14) )
        break;
      v11 = sub_1061B40(*v9, v13);
      if ( !v15 )
      {
        if ( !v11 )
          v9 = 0;
        v15 = v9;
      }
      v12 = v8 & (v16 + i);
      ++v16;
    }
    v5 = 0;
    if ( v15 )
      v9 = v15;
    *a3 = v9;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
  return v5;
}
