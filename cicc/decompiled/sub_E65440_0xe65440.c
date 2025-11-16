// Function: sub_E65440
// Address: 0xe65440
//
_BOOL8 __fastcall sub_E65440(__int64 a1, _QWORD *a2, unsigned __int64 a3)
{
  _BOOL4 v5; // eax
  _BOOL4 v6; // r12d
  int v8; // r15d
  __int64 v9; // rbx
  int v10; // eax
  int v11; // ecx
  int v12; // r8d
  unsigned int i; // r15d
  __int64 v14; // rax
  const void *v15; // rsi
  unsigned int v16; // r15d
  int v17; // eax
  int v18; // [rsp+8h] [rbp-38h]
  int v19; // [rsp+Ch] [rbp-34h]

  v5 = sub_E653E0(a1, (__int64)a2, a3);
  if ( v5 )
    return 1;
  v8 = *(_DWORD *)(a1 + 2464);
  v6 = v5;
  if ( v8 )
  {
    v9 = *(_QWORD *)(a1 + 2448);
    v10 = sub_C94890(a2, a3);
    v11 = v8 - 1;
    v12 = 1;
    for ( i = (v8 - 1) & v10; ; i = v11 & v16 )
    {
      v14 = v9 + 16LL * i;
      v15 = *(const void **)v14;
      if ( *(_QWORD *)v14 == -1 )
        break;
      if ( v15 == (const void *)-2LL )
      {
        if ( a2 == (_QWORD *)-2LL )
          return 1;
      }
      else if ( a3 == *(_QWORD *)(v14 + 8) )
      {
        v18 = v12;
        v19 = v11;
        if ( !a3 )
          return 1;
        v17 = memcmp(a2, v15, a3);
        v11 = v19;
        v12 = v18;
        if ( !v17 )
          return 1;
      }
      v16 = v12 + i;
      ++v12;
    }
    if ( a2 == (_QWORD *)-1LL )
      return 1;
  }
  return v6;
}
