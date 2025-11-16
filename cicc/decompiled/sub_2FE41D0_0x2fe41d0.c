// Function: sub_2FE41D0
// Address: 0x2fe41d0
//
__int64 __fastcall sub_2FE41D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rax
  _QWORD v9[3]; // [rsp+0h] [rbp-20h] BYREF

  v9[0] = a4;
  v9[1] = a5;
  if ( (_WORD)a4 )
  {
    if ( (unsigned __int16)(a4 - 17) > 0xD3u )
      return 1;
  }
  else if ( !(unsigned __int8)sub_30070B0(v9, a2, a3) )
  {
    return 1;
  }
  v6 = *(_QWORD *)(a2 + 56);
  if ( v6 )
  {
    v7 = 1;
    do
    {
      while ( *(_DWORD *)(v6 + 8) )
      {
        v6 = *(_QWORD *)(v6 + 32);
        if ( !v6 )
          goto LABEL_13;
      }
      if ( !v7 )
        return 0;
      v8 = *(_QWORD *)(v6 + 32);
      if ( !v8 )
        return 1;
      if ( !*(_DWORD *)(v8 + 8) )
        return 0;
      v6 = *(_QWORD *)(v8 + 32);
      v7 = 0;
    }
    while ( v6 );
LABEL_13:
    if ( v7 != 1 )
      return 1;
  }
  return 0;
}
