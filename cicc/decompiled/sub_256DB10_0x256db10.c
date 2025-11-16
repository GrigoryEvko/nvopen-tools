// Function: sub_256DB10
// Address: 0x256db10
//
__int64 __fastcall sub_256DB10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  unsigned int v7; // esi
  __int64 v8; // r8
  int v9; // r10d
  __int64 *v10; // rdi
  unsigned int v11; // eax
  __int64 *v12; // rcx
  __int64 v13; // rdx
  int v15; // eax
  int v16; // edx
  __int64 v17; // rcx
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 i; // r15
  _BYTE *v21; // r12
  __int64 v22; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v23; // [rsp+8h] [rbp-38h] BYREF

  v5 = a1 + 360;
  v7 = *(_DWORD *)(a1 + 384);
  v22 = a3;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 360);
    v23 = 0;
LABEL_29:
    v7 *= 2;
    goto LABEL_30;
  }
  v8 = *(_QWORD *)(a1 + 368);
  v9 = 1;
  v10 = 0;
  v11 = (v7 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v12 = (__int64 *)(v8 + 8LL * v11);
  v13 = *v12;
  if ( a3 == *v12 )
    return 0;
  while ( v13 != -4096 )
  {
    if ( v10 || v13 != -8192 )
      v12 = v10;
    v11 = (v7 - 1) & (v9 + v11);
    v13 = *(_QWORD *)(v8 + 8LL * v11);
    if ( a3 == v13 )
      return 0;
    ++v9;
    v10 = v12;
    v12 = (__int64 *)(v8 + 8LL * v11);
  }
  v15 = *(_DWORD *)(a1 + 376);
  if ( !v10 )
    v10 = v12;
  ++*(_QWORD *)(a1 + 360);
  v16 = v15 + 1;
  v23 = v10;
  if ( 4 * (v15 + 1) >= 3 * v7 )
    goto LABEL_29;
  v17 = a3;
  if ( v7 - *(_DWORD *)(a1 + 380) - v16 <= v7 >> 3 )
  {
LABEL_30:
    sub_E3B4A0(v5, v7);
    sub_F9EAB0(v5, &v22, &v23);
    v17 = v22;
    v10 = v23;
    v16 = *(_DWORD *)(a1 + 376) + 1;
  }
  *(_DWORD *)(a1 + 376) = v16;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 380);
  *v10 = v17;
  v18 = *(_QWORD *)(a3 + 56);
  v19 = a3 + 48;
  for ( i = 0x8000000000041LL; v19 != v18; v18 = *(_QWORD *)(v18 + 8) )
  {
    if ( !v18 )
      BUG();
    if ( (unsigned __int8)(*(_BYTE *)(v18 - 24) - 34) <= 0x33u )
    {
      if ( _bittest64(&i, (unsigned int)*(unsigned __int8 *)(v18 - 24) - 34) )
      {
        v21 = *(_BYTE **)(v18 - 56);
        if ( v21 )
        {
          if ( !*v21 && (v21[32] & 0xFu) - 7 <= 1 )
          {
            if ( *(_BYTE *)(a2 + 4299) )
              sub_252C210(a2, *(_QWORD *)(v18 - 56));
            if ( *(_QWORD *)(a2 + 4320) )
              (*(void (__fastcall **)(__int64, __int64, _BYTE *))(a2 + 4328))(a2 + 4304, a2, v21);
          }
        }
      }
    }
  }
  return 1;
}
