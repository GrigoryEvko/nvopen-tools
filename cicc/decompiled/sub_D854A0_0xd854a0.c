// Function: sub_D854A0
// Address: 0xd854a0
//
__int64 __fastcall sub_D854A0(__int64 a1, const void *a2, size_t a3)
{
  unsigned __int64 v3; // rdi
  __int64 *v4; // r14
  __int64 *v5; // rbx
  __int64 v6; // r15
  int v7; // eax
  __int64 v8; // r12
  int v9; // eax
  __int64 v10; // rax
  int v11; // eax
  int v12; // eax
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-40h]
  size_t n; // [rsp+8h] [rbp-38h]

  v3 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v3 )
    return 0;
  v4 = *(__int64 **)(v3 + 32);
  v5 = *(__int64 **)(v3 + 24);
  v15 = (char *)v4 - (char *)v5;
  if ( v4 == v5 )
    return 0;
  v6 = 0;
  do
  {
    while ( 1 )
    {
      v8 = *v5;
      if ( *(char *)(*v5 + 12) >= 0 )
        break;
      v9 = *(_DWORD *)(v8 + 8);
      if ( !v9 )
      {
        v10 = *(_QWORD *)(v8 + 64);
        if ( !v10 )
          break;
        v9 = *(_DWORD *)(v10 + 8);
      }
      if ( v9 != 1 )
        break;
      v11 = *(_BYTE *)(v8 + 12) & 0xF;
      if ( (unsigned int)(v11 - 7) <= 1 )
      {
        if ( *(_QWORD *)(v8 + 32) == a3 )
        {
          if ( !a3 || (n = a3, v7 = memcmp(*(const void **)(v8 + 24), a2, a3), a3 = n, !v7) )
          {
            v6 = v8;
            goto LABEL_18;
          }
        }
        break;
      }
      if ( (*(_BYTE *)(v8 + 12) & 0xF) != 0 && (unsigned int)(v11 - 4) > 1 )
      {
        if ( (((*(_BYTE *)(v8 + 12) & 0xF) + 15) & 0xFu) <= 2 && v15 == 8 )
          v6 = *v5;
        break;
      }
      if ( v6 )
        return 0;
      ++v5;
      v6 = v8;
      if ( v4 == v5 )
        goto LABEL_18;
    }
    ++v5;
  }
  while ( v4 != v5 );
LABEL_18:
  while ( v6 && *(char *)(v6 + 12) < 0 && (*(_BYTE *)(v6 + 13) & 1) != 0 )
  {
    v12 = *(_DWORD *)(v6 + 8);
    if ( v12 == 1 )
      return v6;
    if ( v12 )
      break;
    v14 = *(_QWORD *)(v6 + 64);
    if ( v6 == v14 || !v14 )
      break;
    v6 = *(_QWORD *)(v6 + 64);
  }
  return 0;
}
