// Function: sub_2F25F50
// Address: 0x2f25f50
//
__int64 __fastcall sub_2F25F50(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // r13
  __int64 v8; // rbx
  __int64 *v9; // rax
  char v11; // dl
  int v12; // ecx
  unsigned int v13; // r14d
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  int v17; // edx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rsi
  _DWORD *v20; // rax
  int v21; // ecx
  unsigned __int64 v22; // rax
  int v23; // [rsp+8h] [rbp-38h]
  int v24; // [rsp+8h] [rbp-38h]
  int v25; // [rsp+Ch] [rbp-34h]

  v7 = a3;
  v8 = a4;
  v25 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
  if ( !*(_BYTE *)(a4 + 28) )
    goto LABEL_8;
  v9 = *(__int64 **)(a4 + 8);
  a4 = *(unsigned int *)(a4 + 20);
  a3 = &v9[a4];
  if ( v9 != a3 )
  {
    while ( a2 != *v9 )
    {
      if ( a3 == ++v9 )
        goto LABEL_7;
    }
    return 1;
  }
LABEL_7:
  if ( (unsigned int)a4 >= *(_DWORD *)(v8 + 16) )
  {
LABEL_8:
    sub_C8CC70(v8, a2, (__int64)a3, a4, a5, a6);
    if ( v11 )
      goto LABEL_9;
    return 1;
  }
  *(_DWORD *)(v8 + 20) = a4 + 1;
  *a3 = a2;
  ++*(_QWORD *)v8;
LABEL_9:
  if ( *(_DWORD *)(v8 + 20) - *(_DWORD *)(v8 + 24) != 16 )
  {
    v12 = *(_DWORD *)(a2 + 40);
    if ( (v12 & 0xFFFFFF) != 1 )
    {
      v13 = 1;
      while ( 1 )
      {
        v16 = *(_QWORD *)(a2 + 32) + 40LL * v13;
        v17 = *(_DWORD *)(v16 + 8);
        if ( v17 == v25 )
          goto LABEL_17;
        v23 = *(_DWORD *)(v16 + 8);
        v18 = sub_2EBEE10(*a1, v17);
        v19 = v18;
        if ( !v18 )
          return 0;
        v14 = *(unsigned __int16 *)(v18 + 68);
        v15 = v23;
        if ( (_WORD)v14 == 20 )
        {
          v20 = *(_DWORD **)(v19 + 32);
          if ( (*v20 & 0xFFF00) != 0 || (v20[10] & 0xFFF00) != 0 || (v21 = v20[12], v21 >= 0) )
          {
LABEL_23:
            if ( *(_DWORD *)v7 != v15 && *(_DWORD *)v7 )
              return 0;
            *(_DWORD *)v7 = v15;
            goto LABEL_16;
          }
          v24 = v20[12];
          v22 = sub_2EBEE10(*a1, v21);
          v19 = v22;
          if ( !v22 )
            return 0;
          v14 = *(unsigned __int16 *)(v22 + 68);
          v15 = v24;
        }
        if ( v14 && v14 != 68 )
          goto LABEL_23;
        if ( !(unsigned __int8)sub_2F25F50(a1, v19, v7, v8) )
          return 0;
LABEL_16:
        v12 = *(_DWORD *)(a2 + 40);
LABEL_17:
        v13 += 2;
        if ( (v12 & 0xFFFFFF) == v13 )
          return 1;
      }
    }
    return 1;
  }
  return 0;
}
