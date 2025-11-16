// Function: sub_15CE0F0
// Address: 0x15ce0f0
//
__int64 __fastcall sub_15CE0F0(__int64 a1)
{
  int v2; // r13d
  __int64 result; // rax
  __int64 *v4; // rbx
  __int64 v5; // rdx
  __int64 *v6; // r12
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // r15
  int v11; // edx
  int v12; // ebx
  unsigned int v13; // r13d
  unsigned int v14; // eax

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
  }
  v4 = *(__int64 **)(a1 + 8);
  result = (unsigned int)(4 * v2);
  v5 = *(unsigned int *)(a1 + 24);
  v6 = &v4[2 * v5];
  if ( (unsigned int)result < 0x40 )
    result = 64;
  if ( (unsigned int)v5 <= (unsigned int)result )
  {
    while ( v4 != v6 )
    {
      result = *v4;
      if ( *v4 != -8 )
      {
        if ( result != -16 )
        {
          v7 = v4[1];
          if ( v7 )
          {
            v8 = *(_QWORD *)(v7 + 24);
            if ( v8 )
              j_j___libc_free_0(v8, *(_QWORD *)(v7 + 40) - v8);
            result = j_j___libc_free_0(v7, 56);
          }
        }
        *v4 = -8;
      }
      v4 += 2;
    }
    goto LABEL_16;
  }
  do
  {
    while ( *v4 == -16 || *v4 == -8 )
    {
LABEL_21:
      v4 += 2;
      if ( v4 == v6 )
        goto LABEL_26;
    }
    v10 = v4[1];
    if ( v10 )
    {
      v9 = *(_QWORD *)(v10 + 24);
      if ( v9 )
        j_j___libc_free_0(v9, *(_QWORD *)(v10 + 40) - v9);
      j_j___libc_free_0(v10, 56);
      goto LABEL_21;
    }
    v4 += 2;
  }
  while ( v4 != v6 );
LABEL_26:
  v11 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
  {
    if ( v11 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_36;
    }
    return (__int64)sub_15CE0B0(a1);
  }
  v12 = 64;
  v13 = v2 - 1;
  if ( v13 )
  {
    _BitScanReverse(&v14, v13);
    v12 = 1 << (33 - (v14 ^ 0x1F));
    if ( v12 < 64 )
      v12 = 64;
  }
  if ( v11 == v12 )
    return (__int64)sub_15CE0B0(a1);
  j___libc_free_0(*(_QWORD *)(a1 + 8));
  result = sub_1454B60(4 * v12 / 3u + 1);
  *(_DWORD *)(a1 + 24) = result;
  if ( (_DWORD)result )
  {
    *(_QWORD *)(a1 + 8) = sub_22077B0(16LL * (unsigned int)result);
    return (__int64)sub_15CE0B0(a1);
  }
LABEL_36:
  *(_QWORD *)(a1 + 8) = 0;
LABEL_16:
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
