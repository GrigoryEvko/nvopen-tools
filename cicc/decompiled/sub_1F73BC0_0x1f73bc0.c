// Function: sub_1F73BC0
// Address: 0x1f73bc0
//
__int64 __fastcall sub_1F73BC0(__int64 a1, __int64 *a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  char *v10; // rdx
  unsigned __int8 v11; // al
  const void **v12; // r14
  __int64 v13; // rdx
  unsigned int v14; // r15d
  __int64 v15; // rcx
  int v16; // edx
  __int64 result; // rax
  __int64 v18; // rcx
  __int64 v19; // rsi
  unsigned int *v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // [rsp+8h] [rbp-58h]
  unsigned int *v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+18h] [rbp-48h] BYREF
  __int64 v26; // [rsp+20h] [rbp-40h] BYREF
  int v27; // [rsp+28h] [rbp-38h]

  v25 = sub_1560340((_QWORD *)(*(_QWORD *)a2[4] + 112LL), -1, "strict-float-cast-overflow", 0x1Au);
  v8 = sub_155D8B0(&v25);
  if ( v9 == 5 && *(_DWORD *)v8 == 1936482662 && *(_BYTE *)(v8 + 4) == 101 )
    return 0;
  v10 = *(char **)(a1 + 40);
  v11 = *v10;
  v12 = (const void **)*((_QWORD *)v10 + 1);
  v13 = 1;
  v14 = v11;
  if ( v11 != 1 )
  {
    if ( !v11 )
      return 0;
    v13 = v11;
    if ( !*(_QWORD *)(a3 + 8LL * v11 + 120) )
      return 0;
  }
  if ( *(_BYTE *)(a3 + 259 * v13 + 2597) || (*(_BYTE *)(*a2 + 792) & 0x20) == 0 )
    return 0;
  v15 = **(_QWORD **)(a1 + 32);
  v16 = *(unsigned __int16 *)(a1 + 24);
  if ( v16 == 146 )
  {
    if ( *(_WORD *)(v15 + 24) == 152 )
    {
      v18 = *(_QWORD *)(v15 + 32);
      if ( v11 == *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v18 + 40LL) + 16LL * *(unsigned int *)(v18 + 8)) )
      {
        v19 = *(_QWORD *)(a1 + 72);
        v26 = v19;
        if ( v19 )
        {
          v22 = v18;
          sub_1623A60((__int64)&v26, v19, 2);
          v18 = v22;
        }
        v27 = *(_DWORD *)(a1 + 64);
        result = sub_1D309E0(a2, 175, (__int64)&v26, v14, v12, 0, a4, a5, a6, *(_OWORD *)v18);
        if ( v26 )
          goto LABEL_26;
        return result;
      }
    }
    return 0;
  }
  if ( v16 != 147 )
    return 0;
  if ( *(_WORD *)(v15 + 24) != 153 )
    return 0;
  v20 = *(unsigned int **)(v15 + 32);
  if ( v11 != *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v20 + 40LL) + 16LL * v20[2]) )
    return 0;
  v21 = *(_QWORD *)(a1 + 72);
  v26 = v21;
  if ( v21 )
  {
    v23 = v20;
    sub_1623A60((__int64)&v26, v21, 2);
    v20 = v23;
  }
  v27 = *(_DWORD *)(a1 + 64);
  result = sub_1D309E0(a2, 175, (__int64)&v26, v14, v12, 0, a4, a5, a6, *(_OWORD *)v20);
  if ( v26 )
  {
LABEL_26:
    v24 = result;
    sub_161E7C0((__int64)&v26, v26);
    return v24;
  }
  return result;
}
