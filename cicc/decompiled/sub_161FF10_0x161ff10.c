// Function: sub_161FF10
// Address: 0x161ff10
//
__int64 __fastcall sub_161FF10(__int64 *a1, void *a2, size_t a3)
{
  __int64 v4; // rbx
  unsigned int v5; // r12d
  __int64 *v6; // r13
  __int64 v7; // rax
  __int64 *v9; // rax
  __int64 v10; // rdx

  v4 = *a1;
  v5 = sub_16D19C0(*a1 + 272, a2, a3);
  v6 = (__int64 *)(*(_QWORD *)(v4 + 272) + 8LL * v5);
  v7 = *v6;
  if ( *v6 )
  {
    if ( v7 != -8 )
      return v7 + 8;
    --*(_DWORD *)(v4 + 288);
  }
  *v6 = (__int64)sub_161FD00(a2, a3, (__int64 *)(v4 + 296));
  ++*(_DWORD *)(v4 + 284);
  v9 = (__int64 *)(*(_QWORD *)(v4 + 272) + 8LL * (unsigned int)sub_16D1CD0(v4 + 272, v5));
  v10 = *v9;
  if ( *v9 != -8 )
    goto LABEL_7;
  do
  {
    do
    {
      v10 = v9[1];
      ++v9;
    }
    while ( v10 == -8 );
LABEL_7:
    ;
  }
  while ( !v10 );
  *(_QWORD *)(v10 + 16) = v10;
  return v10 + 8;
}
