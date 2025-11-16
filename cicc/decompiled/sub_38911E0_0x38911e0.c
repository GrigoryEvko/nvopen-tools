// Function: sub_38911E0
// Address: 0x38911e0
//
__int64 __fastcall sub_38911E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rbx
  size_t v4; // r12
  size_t v5; // r13
  size_t v6; // rdx
  int v7; // eax
  unsigned int v8; // r12d
  __int64 v9; // rax
  __int64 v10; // rdx
  size_t v11; // rbx
  size_t v12; // r12
  size_t v13; // rdx
  int v14; // eax
  __int64 v15; // rbx
  __int64 v17; // r12

  v2 = a1 + 8;
  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
    return a1 + 8;
  while ( 1 )
  {
    if ( *(_DWORD *)(v3 + 32) <= 1u )
    {
      LOBYTE(v8) = *(_DWORD *)(v3 + 48) < *(_DWORD *)(a2 + 16);
      goto LABEL_8;
    }
    v4 = *(_QWORD *)(v3 + 72);
    v5 = *(_QWORD *)(a2 + 40);
    v6 = v5;
    if ( v4 <= v5 )
      v6 = *(_QWORD *)(v3 + 72);
    if ( v6 )
    {
      v7 = memcmp(*(const void **)(v3 + 64), *(const void **)(a2 + 32), v6);
      if ( v7 )
      {
        LOBYTE(v8) = v7 < 0;
        goto LABEL_8;
      }
    }
    v17 = v4 - v5;
    if ( v17 < 0x80000000LL )
      break;
    v9 = *(_QWORD *)(v3 + 16);
    v2 = v3;
LABEL_10:
    if ( !v9 )
      goto LABEL_15;
LABEL_11:
    v3 = v9;
  }
  if ( v17 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
  {
    v10 = *(_QWORD *)(v3 + 24);
    goto LABEL_9;
  }
  v8 = (unsigned int)v17 >> 31;
LABEL_8:
  v9 = *(_QWORD *)(v3 + 16);
  v10 = *(_QWORD *)(v3 + 24);
  if ( (_BYTE)v8 )
  {
LABEL_9:
    v9 = v10;
    goto LABEL_10;
  }
  v2 = v3;
  if ( v9 )
    goto LABEL_11;
LABEL_15:
  if ( a1 + 8 != v2 )
  {
    if ( *(_DWORD *)a2 <= 1u )
    {
      if ( *(_DWORD *)(a2 + 16) < *(_DWORD *)(v2 + 48) )
        return a1 + 8;
    }
    else
    {
      v11 = *(_QWORD *)(a2 + 40);
      v12 = *(_QWORD *)(v2 + 72);
      v13 = v12;
      if ( v11 <= v12 )
        v13 = *(_QWORD *)(a2 + 40);
      if ( v13 && (v14 = memcmp(*(const void **)(a2 + 32), *(const void **)(v2 + 64), v13)) != 0 )
      {
LABEL_24:
        if ( v14 < 0 )
          return a1 + 8;
      }
      else
      {
        v15 = v11 - v12;
        if ( v15 <= 0x7FFFFFFF )
        {
          if ( v15 >= (__int64)0xFFFFFFFF80000000LL )
          {
            v14 = v15;
            goto LABEL_24;
          }
          return a1 + 8;
        }
      }
    }
  }
  return v2;
}
