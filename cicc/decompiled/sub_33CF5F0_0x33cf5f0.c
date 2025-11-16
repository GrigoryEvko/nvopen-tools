// Function: sub_33CF5F0
// Address: 0x33cf5f0
//
__int64 __fastcall sub_33CF5F0(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rsi
  _DWORD *v4; // rdi
  __int64 v5; // r9
  int v6; // edx
  __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // rax

  v3 = a1;
  if ( *(_DWORD *)(a1 + 24) != 234 )
    return a1;
  while ( 1 )
  {
    v4 = *(_DWORD **)(v3 + 40);
    v5 = v3;
    v3 = *(_QWORD *)v4;
    v6 = v4[2];
    v7 = *(_QWORD *)(*(_QWORD *)v4 + 56LL);
    if ( !v7 )
      return v5;
    v8 = 1;
    do
    {
      while ( v6 != *(_DWORD *)(v7 + 8) )
      {
        v7 = *(_QWORD *)(v7 + 32);
        if ( !v7 )
          goto LABEL_10;
      }
      if ( !v8 )
        return v5;
      v9 = *(_QWORD *)(v7 + 32);
      if ( !v9 )
        goto LABEL_11;
      if ( v6 == *(_DWORD *)(v9 + 8) )
        return v5;
      v7 = *(_QWORD *)(v9 + 32);
      v8 = 0;
    }
    while ( v7 );
LABEL_10:
    if ( v8 == 1 )
      return v5;
LABEL_11:
    a2 = (unsigned int)v4[2] | a2 & 0xFFFFFFFF00000000LL;
    if ( *(_DWORD *)(v3 + 24) != 234 )
      return *(_QWORD *)v4;
  }
}
