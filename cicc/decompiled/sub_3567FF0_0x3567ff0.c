// Function: sub_3567FF0
// Address: 0x3567ff0
//
__int64 __fastcall sub_3567FF0(_QWORD *a1)
{
  __int64 v1; // r15
  unsigned __int64 v2; // rax
  __int64 *v3; // rbx
  __int64 *v4; // r13
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned int i; // eax

  v1 = 0;
  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(__int64 **)(v2 + 64);
  v4 = &v3[*(unsigned int *)(v2 + 72)];
  if ( v3 == v4 )
    return 0;
  v5 = *v3;
  v6 = a1[3];
  if ( !*v3 )
    goto LABEL_14;
LABEL_3:
  v7 = (unsigned int)(*(_DWORD *)(v5 + 24) + 1);
  for ( i = *(_DWORD *)(v5 + 24) + 1; ; i = 0 )
  {
    if ( i < *(_DWORD *)(v6 + 32)
      && *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v7)
      && (unsigned __int8)sub_3567D90(a1, v5) != 1
      && v5 )
    {
      if ( v1 )
        return 0;
      v1 = v5;
    }
    if ( v4 == ++v3 )
      break;
    v5 = *v3;
    v6 = a1[3];
    if ( *v3 )
      goto LABEL_3;
LABEL_14:
    v7 = 0;
  }
  return v1;
}
