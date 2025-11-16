// Function: sub_B8E0B0
// Address: 0xb8e0b0
//
bool __fastcall sub_B8E0B0(__int64 *a1, unsigned int a2)
{
  __int64 v2; // rdx
  _DWORD *v3; // rax
  __int64 v4; // rdx
  bool result; // al
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx

  v2 = *a1;
  if ( *(_QWORD *)(*a1 + 184) )
  {
    v6 = *(_QWORD *)(v2 + 160);
    v7 = v2 + 152;
    if ( !v6 )
      return 1;
    v8 = v2 + 152;
    do
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(v6 + 16);
        v10 = *(_QWORD *)(v6 + 24);
        if ( a2 <= *(_DWORD *)(v6 + 32) )
          break;
        v6 = *(_QWORD *)(v6 + 24);
        if ( !v10 )
          goto LABEL_13;
      }
      v8 = v6;
      v6 = *(_QWORD *)(v6 + 16);
    }
    while ( v9 );
LABEL_13:
    result = 1;
    if ( v7 != v8 )
      return a2 < *(_DWORD *)(v8 + 32);
  }
  else
  {
    v3 = *(_DWORD **)v2;
    v4 = *(_QWORD *)v2 + 4LL * *(unsigned int *)(v2 + 8);
    if ( v3 == (_DWORD *)v4 )
      return 1;
    while ( a2 != *v3 )
    {
      if ( (_DWORD *)v4 == ++v3 )
        return 1;
    }
    return v3 == (_DWORD *)v4;
  }
  return result;
}
