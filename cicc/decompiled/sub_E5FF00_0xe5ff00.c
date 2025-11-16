// Function: sub_E5FF00
// Address: 0xe5ff00
//
__int64 __fastcall sub_E5FF00(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  __int64 v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx

  v2 = *(_QWORD *)(a1 + 224);
  v3 = a1 + 216;
  if ( !v2 )
    return -1;
  v4 = a1 + 216;
  do
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(v2 + 16);
      v6 = *(_QWORD *)(v2 + 24);
      if ( *(_DWORD *)(v2 + 32) >= a2 )
        break;
      v2 = *(_QWORD *)(v2 + 24);
      if ( !v6 )
        goto LABEL_6;
    }
    v4 = v2;
    v2 = *(_QWORD *)(v2 + 16);
  }
  while ( v5 );
LABEL_6:
  if ( v3 != v4 && *(_DWORD *)(v4 + 32) <= a2 )
    return *(_QWORD *)(v4 + 40);
  else
    return -1;
}
