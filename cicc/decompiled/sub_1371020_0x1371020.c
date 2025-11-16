// Function: sub_1371020
// Address: 0x1371020
//
__int64 __fastcall sub_1371020(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  size_t v5; // rdx
  __int64 result; // rax
  __int64 v7; // rdi
  _DWORD *v8; // r8
  _DWORD *v9; // rcx
  __int64 v10; // rsi
  __int64 *v11; // rax
  __int64 *v12; // rdx

  v3 = *(unsigned int *)(a2 + 136);
  *(_DWORD *)(a2 + 24) = 0;
  v5 = 8 * v3;
  if ( v5 )
    memset(*(void **)(a2 + 128), 0, v5);
  result = *(_QWORD *)(a2 + 96);
  v7 = result + 4;
  v8 = (_DWORD *)(result + 4LL * *(unsigned int *)(a2 + 104));
  if ( (_DWORD *)(result + 4) != v8 )
  {
    v9 = (_DWORD *)(result + 4);
    do
    {
      v10 = *(_QWORD *)(a1 + 64) + 24LL * (unsigned int)*v9;
      v11 = *(__int64 **)(v10 + 8);
      if ( !v11 || !*((_BYTE *)v11 + 8) )
        goto LABEL_14;
      do
      {
        v12 = v11;
        v11 = (__int64 *)*v11;
      }
      while ( v11 && *((_BYTE *)v11 + 8) );
      if ( *(_DWORD *)v10 == *(_DWORD *)v12[12] )
      {
LABEL_14:
        v7 += 4;
        *(_DWORD *)(v7 - 4) = *v9;
      }
      ++v9;
    }
    while ( v9 != v8 );
    result = *(_QWORD *)(a2 + 96);
  }
  *(_DWORD *)(a2 + 104) = (v7 - result) >> 2;
  return result;
}
