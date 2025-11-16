// Function: sub_14CDF70
// Address: 0x14cdf70
//
void __fastcall sub_14CDF70(__int64 a1)
{
  __int64 v1; // r13
  __int64 j; // rbx
  __int64 v3; // rax
  unsigned int v4; // eax
  __int64 v5; // r12
  __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 i; // [rsp+0h] [rbp-60h]
  _QWORD v13[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+20h] [rbp-40h]
  int v15; // [rsp+28h] [rbp-38h]

  v1 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  for ( i = *(_QWORD *)a1 + 72LL; i != v1; v1 = *(_QWORD *)(v1 + 8) )
  {
    if ( !v1 )
      JUMPOUT(0x41974E);
    for ( j = *(_QWORD *)(v1 + 24); v1 + 16 != j; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
        BUG();
      if ( *(_BYTE *)(j - 8) == 78 )
      {
        v3 = *(_QWORD *)(j - 48);
        if ( !*(_BYTE *)(v3 + 16) && *(_DWORD *)(v3 + 36) == 4 )
        {
          v13[0] = 6;
          v13[1] = 0;
          v14 = j - 24;
          if ( j != 8 && j != 16 )
            sub_164C220(v13);
          v15 = -1;
          v4 = *(_DWORD *)(a1 + 16);
          if ( v4 >= *(_DWORD *)(a1 + 20) )
          {
            sub_14CB640(a1 + 8, 0);
            v4 = *(_DWORD *)(a1 + 16);
          }
          v5 = *(_QWORD *)(a1 + 8) + 32LL * v4;
          if ( v5 )
          {
            *(_QWORD *)v5 = 6;
            *(_QWORD *)(v5 + 8) = 0;
            v6 = v14;
            v7 = v14 == 0;
            *(_QWORD *)(v5 + 16) = v14;
            if ( v6 != -8 && !v7 && v6 != -16 )
              sub_1649AC0(v5, v13[0] & 0xFFFFFFFFFFFFFFF8LL);
            *(_DWORD *)(v5 + 24) = v15;
            v4 = *(_DWORD *)(a1 + 16);
          }
          *(_DWORD *)(a1 + 16) = v4 + 1;
          if ( v14 != -8 && v14 != 0 && v14 != -16 )
            sub_1649B30(v13);
        }
      }
    }
  }
  v8 = *(unsigned int *)(a1 + 16);
  v9 = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(a1 + 184) = 1;
  v10 = v9 + 32 * v8;
  while ( v9 != v10 )
  {
    v11 = *(_QWORD *)(v9 + 16);
    v9 += 32;
    sub_14CDA00(a1, v11);
  }
}
