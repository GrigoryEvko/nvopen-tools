// Function: sub_3434600
// Address: 0x3434600
//
__int64 __fastcall sub_3434600(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx

  v3 = a1 + 8;
  v4 = *(_QWORD *)(a1 + 16);
  if ( !v4 )
    return v3;
  v5 = *a2;
  v6 = v3;
  do
  {
    while ( *(_QWORD *)(v4 + 32) >= v5 && (*(_QWORD *)(v4 + 32) != v5 || *(_DWORD *)(v4 + 40) >= *((_DWORD *)a2 + 2)) )
    {
      v6 = v4;
      v4 = *(_QWORD *)(v4 + 16);
      if ( !v4 )
        goto LABEL_8;
    }
    v4 = *(_QWORD *)(v4 + 24);
  }
  while ( v4 );
LABEL_8:
  if ( v3 == v6 || v5 < *(_QWORD *)(v6 + 32) || v5 == *(_QWORD *)(v6 + 32) && *((_DWORD *)a2 + 2) < *(_DWORD *)(v6 + 40) )
    return v3;
  else
    return v6;
}
