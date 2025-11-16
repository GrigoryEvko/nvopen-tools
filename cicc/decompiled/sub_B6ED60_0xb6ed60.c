// Function: sub_B6ED60
// Address: 0xb6ed60
//
__int64 __fastcall sub_B6ED60(__int64 *a1, const void *a2, size_t a3)
{
  __int64 v4; // r12
  unsigned int v5; // eax
  unsigned int v6; // r14d
  __int64 *v7; // r9
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 *v11; // r9
  __int64 v12; // rcx
  __int64 *v13; // rax
  __int64 *v14; // rax
  __int64 v15; // [rsp+8h] [rbp-68h]
  __int64 *v16; // [rsp+10h] [rbp-60h]
  int v17; // [rsp+1Ch] [rbp-54h]

  v4 = *a1;
  v17 = *(_DWORD *)(*a1 + 3212);
  v5 = sub_C92610(a2, a3);
  v6 = sub_C92740(v4 + 3200, a2, a3, v5);
  v7 = (__int64 *)(*(_QWORD *)(v4 + 3200) + 8LL * v6);
  v8 = *v7;
  if ( *v7 )
  {
    if ( v8 != -8 )
      return *(unsigned int *)(v8 + 8);
    --*(_DWORD *)(v4 + 3216);
  }
  v16 = v7;
  v10 = sub_C7D670(a3 + 17, 8);
  v11 = v16;
  v12 = v10;
  if ( a3 )
  {
    v15 = v10;
    memcpy((void *)(v10 + 16), a2, a3);
    v11 = v16;
    v12 = v15;
  }
  *(_BYTE *)(v12 + a3 + 16) = 0;
  *(_QWORD *)v12 = a3;
  *(_DWORD *)(v12 + 8) = v17;
  *v11 = v12;
  ++*(_DWORD *)(v4 + 3212);
  v13 = (__int64 *)(*(_QWORD *)(v4 + 3200) + 8LL * (unsigned int)sub_C929D0(v4 + 3200, v6));
  v8 = *v13;
  if ( !*v13 || v8 == -8 )
  {
    v14 = v13 + 1;
    do
    {
      do
        v8 = *v14++;
      while ( !v8 );
    }
    while ( v8 == -8 );
  }
  return *(unsigned int *)(v8 + 8);
}
