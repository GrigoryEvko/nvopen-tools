// Function: sub_1AE1280
// Address: 0x1ae1280
//
__int64 __fastcall sub_1AE1280(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rdi
  __int64 i; // r15
  __int64 v6; // rbx
  __int64 j; // r12
  __int64 v9[2]; // [rsp+10h] [rbp-50h] BYREF
  char v10; // [rsp+20h] [rbp-40h]
  char v11; // [rsp+21h] [rbp-3Fh]

  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, a2);
    v2 = *(_QWORD *)(a2 + 88);
    v3 = v2 + 40LL * *(_QWORD *)(a2 + 96);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2, a2);
      v2 = *(_QWORD *)(a2 + 88);
    }
  }
  else
  {
    v2 = *(_QWORD *)(a2 + 88);
    v3 = v2 + 40LL * *(_QWORD *)(a2 + 96);
  }
  while ( v3 != v2 )
  {
    while ( (*(_BYTE *)(v2 + 23) & 0x20) != 0 )
    {
      v2 += 40;
      if ( v3 == v2 )
        goto LABEL_8;
    }
    v4 = v2;
    v2 += 40;
    v11 = 1;
    v9[0] = (__int64)"arg";
    v10 = 3;
    sub_164B780(v4, v9);
  }
LABEL_8:
  for ( i = *(_QWORD *)(a2 + 80); a2 + 72 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( (*(_BYTE *)(i - 1) & 0x20) == 0 )
    {
      v11 = 1;
      v9[0] = (__int64)"bb";
      v10 = 3;
      sub_164B780(i - 24, v9);
    }
    v6 = *(_QWORD *)(i + 24);
    for ( j = i + 16; j != v6; v6 = *(_QWORD *)(v6 + 8) )
    {
      while ( 1 )
      {
        if ( !v6 )
          BUG();
        if ( (*(_BYTE *)(v6 - 1) & 0x20) == 0 && *(_BYTE *)(*(_QWORD *)(v6 - 24) + 8LL) )
          break;
        v6 = *(_QWORD *)(v6 + 8);
        if ( j == v6 )
          goto LABEL_19;
      }
      v11 = 1;
      v9[0] = (__int64)"tmp";
      v10 = 3;
      sub_164B780(v6 - 24, v9);
    }
LABEL_19:
    ;
  }
  return 1;
}
