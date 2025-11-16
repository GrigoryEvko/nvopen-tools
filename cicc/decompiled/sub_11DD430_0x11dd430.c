// Function: sub_11DD430
// Address: 0x11dd430
//
char *__fastcall sub_11DD430(__int64 *a1, unsigned int a2)
{
  __int64 v2; // rdi
  __int64 v4; // rax
  __int64 v5; // r8
  unsigned int v6; // ecx
  int *v7; // rdx
  int v8; // edi
  int v9; // edx
  int v10; // r10d

  if ( (a1[((unsigned __int64)a2 >> 6) + 1] & (1LL << a2)) != 0 )
    return 0;
  v2 = *a1;
  if ( (((int)*(unsigned __int8 *)(v2 + (a2 >> 2)) >> (2 * (a2 & 3))) & 3) == 0 )
    return 0;
  if ( (((int)*(unsigned __int8 *)(v2 + (a2 >> 2)) >> (2 * (a2 & 3))) & 3) == 3 )
    return (&off_4977320)[2 * a2];
  v4 = *(unsigned int *)(v2 + 160);
  v5 = *(_QWORD *)(v2 + 144);
  if ( !(_DWORD)v4 )
    goto LABEL_9;
  v6 = (v4 - 1) & (37 * a2);
  v7 = (int *)(v5 + 40LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v9 = 1;
    while ( v8 != -1 )
    {
      v10 = v9 + 1;
      v6 = (v4 - 1) & (v9 + v6);
      v7 = (int *)(v5 + 40LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        return (char *)*((_QWORD *)v7 + 1);
      v9 = v10;
    }
LABEL_9:
    v7 = (int *)(v5 + 40 * v4);
  }
  return (char *)*((_QWORD *)v7 + 1);
}
