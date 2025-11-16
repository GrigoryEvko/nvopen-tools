// Function: sub_E70000
// Address: 0xe70000
//
bool __fastcall sub_E70000(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // r12
  __int64 v5; // rdi
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  bool result; // al
  unsigned int v11; // [rsp+4h] [rbp-1Ch] BYREF
  unsigned int *v12; // [rsp+8h] [rbp-18h] BYREF

  v3 = a2;
  v5 = a1 + 1736;
  v6 = v5;
  v7 = *(_QWORD *)(a1 + 1744);
  v11 = a3;
  if ( !v7 )
    goto LABEL_11;
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v7 + 16);
      v9 = *(_QWORD *)(v7 + 24);
      if ( a3 <= *(_DWORD *)(v7 + 32) )
        break;
      v7 = *(_QWORD *)(v7 + 24);
      if ( !v9 )
        goto LABEL_6;
    }
    v6 = v7;
    v7 = *(_QWORD *)(v7 + 16);
  }
  while ( v8 );
LABEL_6:
  if ( v5 == v6 || a3 < *(_DWORD *)(v6 + 32) )
  {
LABEL_11:
    v12 = &v11;
    v6 = sub_E6FC80((_QWORD *)(a1 + 1728), v6, &v12);
    if ( !(_DWORD)v3 )
      return *(_WORD *)(a1 + 1904) > 4u;
  }
  else if ( !(_DWORD)v3 )
  {
    return *(_WORD *)(a1 + 1904) > 4u;
  }
  result = 0;
  if ( (unsigned int)v3 < *(_DWORD *)(v6 + 168) )
    return *(_QWORD *)(*(_QWORD *)(v6 + 160) + 80 * v3 + 8) != 0;
  return result;
}
