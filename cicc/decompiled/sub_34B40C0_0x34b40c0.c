// Function: sub_34B40C0
// Address: 0x34b40c0
//
__int64 __fastcall sub_34B40C0(__int64 a1, unsigned int *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rbx
  unsigned int v5; // edx
  __int64 v6; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r12

  v3 = *(_QWORD *)(a1 + 16);
  v4 = a1 + 8;
  if ( !v3 )
    return 0;
  v5 = *a2;
  while ( 1 )
  {
    while ( *(_DWORD *)(v3 + 32) < v5 )
    {
      v3 = *(_QWORD *)(v3 + 24);
      if ( !v3 )
        return 0;
    }
    v6 = *(_QWORD *)(v3 + 16);
    if ( *(_DWORD *)(v3 + 32) <= v5 )
      break;
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    if ( !v6 )
      return 0;
  }
  v8 = *(_QWORD *)(v3 + 24);
  if ( v8 )
  {
    do
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(v8 + 16);
        v10 = *(_QWORD *)(v8 + 24);
        if ( v5 < *(_DWORD *)(v8 + 32) )
          break;
        v8 = *(_QWORD *)(v8 + 24);
        if ( !v10 )
          goto LABEL_13;
      }
      v4 = v8;
      v8 = *(_QWORD *)(v8 + 16);
    }
    while ( v9 );
  }
LABEL_13:
  while ( v6 )
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v6 + 24);
      if ( v5 <= *(_DWORD *)(v6 + 32) )
        break;
      v6 = *(_QWORD *)(v6 + 24);
      if ( !v11 )
        goto LABEL_16;
    }
    v3 = v6;
    v6 = *(_QWORD *)(v6 + 16);
  }
LABEL_16:
  if ( v4 == v3 )
    return 0;
  v12 = 0;
  do
  {
    ++v12;
    v3 = sub_220EF30(v3);
  }
  while ( v3 != v4 );
  return v12;
}
