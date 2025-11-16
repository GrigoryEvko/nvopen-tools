// Function: sub_E6B3F0
// Address: 0xe6b3f0
//
unsigned __int64 __fastcall sub_E6B3F0(__int64 a1, const void *a2, size_t a3)
{
  __int64 *v4; // r13
  int v6; // eax
  unsigned int v7; // r8d
  unsigned __int64 *v8; // r9
  unsigned __int64 result; // rax
  size_t *v10; // rdi
  size_t v11; // rax
  unsigned __int64 v12; // r12
  size_t v13; // rdx
  _BYTE *v14; // rcx
  unsigned __int64 *v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // rax
  unsigned __int64 *v18; // [rsp+0h] [rbp-40h]
  unsigned __int64 *v19; // [rsp+0h] [rbp-40h]
  unsigned int v20; // [rsp+Ch] [rbp-34h]
  unsigned int v21; // [rsp+Ch] [rbp-34h]

  v4 = (__int64 *)(a1 + 1344);
  v6 = sub_C92610();
  v7 = sub_C92740(a1 + 1344, a2, a3, v6);
  v8 = (unsigned __int64 *)(*(_QWORD *)(a1 + 1344) + 8LL * v7);
  result = *v8;
  if ( *v8 )
  {
    if ( result != -8 )
      return result;
    --*(_DWORD *)(a1 + 1360);
  }
  v10 = *(size_t **)(a1 + 1368);
  v11 = *v10;
  v10[10] += a3 + 25;
  v12 = (v11 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v13 = a3 + 25 + v12;
  if ( v10[1] < v13 || !v11 )
  {
    v19 = v8;
    v21 = v7;
    v17 = sub_9D1E70((__int64)v10, a3 + 25, a3 + 25, 3);
    v7 = v21;
    v8 = v19;
    v12 = v17;
    v14 = (_BYTE *)(v17 + 24);
    if ( !a3 )
    {
      *(_BYTE *)(v17 + 24) = 0;
      goto LABEL_9;
    }
LABEL_15:
    v18 = v8;
    v20 = v7;
    v16 = memcpy(v14, a2, a3);
    v8 = v18;
    v7 = v20;
    v16[a3] = 0;
    if ( !v12 )
      goto LABEL_10;
    goto LABEL_9;
  }
  *v10 = v13;
  v14 = (_BYTE *)(v12 + 24);
  if ( a3 )
    goto LABEL_15;
  *v14 = 0;
  if ( v12 )
  {
LABEL_9:
    *(_QWORD *)v12 = a3;
    *(_QWORD *)(v12 + 8) = 0;
    *(_DWORD *)(v12 + 16) = 0;
    *(_BYTE *)(v12 + 20) = 0;
  }
LABEL_10:
  *v8 = v12;
  ++*(_DWORD *)(a1 + 1356);
  v15 = (unsigned __int64 *)(*(_QWORD *)(a1 + 1344) + 8LL * (unsigned int)sub_C929D0(v4, v7));
  result = *v15;
  if ( !*v15 || result == -8 )
  {
    do
    {
      do
      {
        result = v15[1];
        ++v15;
      }
      while ( result == -8 );
    }
    while ( !result );
  }
  return result;
}
