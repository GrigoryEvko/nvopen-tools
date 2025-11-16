// Function: sub_8C78D0
// Address: 0x8c78d0
//
__int64 __fastcall sub_8C78D0(__int64 a1)
{
  _BOOL4 v1; // r13d
  __int64 *v2; // rax
  __int64 v3; // r14
  __int64 v4; // r12
  __int128 *v6; // r13
  unsigned int *v7; // rsi
  __int64 *v8; // rdi
  __int64 **v9; // rax
  __int64 *v10; // rsi

  v1 = 1;
  v2 = *(__int64 **)(a1 + 32);
  if ( !v2 )
    return v1;
  v3 = *v2;
  v4 = a1;
  if ( a1 != *v2 || (v4 = v2[1]) != 0 && v3 != v4 )
  {
    v1 = sub_8C7610(v4);
    if ( v1
      && (!(unsigned int)sub_8DED30(*(_QWORD *)(v4 + 120), *(_QWORD *)(v3 + 120), 261)
       || !(unsigned int)sub_8DBAE0(*(_QWORD *)(v4 + 120), *(_QWORD *)(v3 + 120))
       || *(_BYTE *)(v4 + 137) != *(_BYTE *)(v3 + 137)
       || ((*(_BYTE *)(v3 + 144) ^ *(_BYTE *)(v4 + 144)) & 0x3C) != 0
       || ((*(_BYTE *)(v3 + 88) ^ *(_BYTE *)(v4 + 88)) & 0x73) != 0) )
    {
      if ( !*(_QWORD *)v4 || (v6 = *(__int128 **)v4, v6 == sub_87EA80()) )
      {
        v8 = *(__int64 **)(*(_QWORD *)(v4 + 40) + 32LL);
        v9 = (__int64 **)v8[4];
        v10 = v8;
        if ( v9 )
          v10 = *v9;
        v7 = (unsigned int *)(v10 + 8);
      }
      else
      {
        v7 = (unsigned int *)(v3 + 64);
        v8 = (__int64 *)v4;
      }
      sub_8C6700(v8, v7, 0x42Au, 0x425u);
      v1 = 0;
      sub_8C7090(8, v4);
    }
    sub_8C6CA0(v4, v3, 8u, (_QWORD *)(v3 + 64));
    sub_8C6CA0(v3, v4, 8u, (_QWORD *)(v4 + 64));
    return v1;
  }
  return 1;
}
