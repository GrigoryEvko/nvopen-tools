// Function: sub_1AA6570
// Address: 0x1aa6570
//
__int64 __fastcall sub_1AA6570(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned int v3; // r12d
  __int64 v4; // rdi
  unsigned __int64 v5; // rbx
  unsigned int i; // r13d
  __int64 v8; // [rsp+0h] [rbp-40h]
  int v9; // [rsp+Ch] [rbp-34h]

  v2 = *(_QWORD *)(a1 + 80);
  v8 = a1 + 72;
  if ( v2 == a1 + 72 )
  {
    return 0;
  }
  else
  {
    v3 = 0;
    do
    {
      while ( 1 )
      {
        v4 = v2 - 24;
        if ( !v2 )
          v4 = 0;
        v5 = sub_157EBA0(v4);
        if ( (unsigned int)sub_15F4D60(v5) > 1 && *(_BYTE *)(v5 + 16) != 28 )
        {
          v9 = sub_15F4D60(v5);
          if ( v9 )
            break;
        }
        v2 = *(_QWORD *)(v2 + 8);
        if ( v8 == v2 )
          return v3;
      }
      for ( i = 0; i != v9; ++i )
        v3 -= (sub_1AAC5F0(v5, i, a2) == 0) - 1;
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v8 != v2 );
  }
  return v3;
}
