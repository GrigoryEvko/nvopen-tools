// Function: sub_B3EE80
// Address: 0xb3ee80
//
__int64 __fastcall sub_B3EE80(__int64 a1, __int64 a2, __int64 **a3)
{
  __int64 result; // rax
  int v5; // r12d
  __int64 v6; // r9
  int v7; // r11d
  __int64 *v8; // r15
  unsigned int i; // r8d
  __int64 *v10; // rcx
  __int64 v11; // r13
  unsigned int v12; // r8d
  size_t v13; // rdx
  int v14; // eax
  size_t v15; // rdx
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  __int64 *v21; // [rsp+10h] [rbp-50h]
  __int64 *v22; // [rsp+18h] [rbp-48h]
  __int64 *v23; // [rsp+18h] [rbp-48h]
  int v24; // [rsp+18h] [rbp-48h]
  int v25; // [rsp+24h] [rbp-3Ch]
  int v26; // [rsp+24h] [rbp-3Ch]
  unsigned int v27; // [rsp+24h] [rbp-3Ch]
  unsigned int v28; // [rsp+28h] [rbp-38h]
  unsigned int v29; // [rsp+28h] [rbp-38h]
  __int64 v30; // [rsp+28h] [rbp-38h]

  result = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)result )
  {
    *a3 = 0;
    return result;
  }
  v5 = result - 1;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  for ( i = (result - 1) & *(_DWORD *)a2; ; i = v5 & v12 )
  {
    v10 = (__int64 *)(v6 + 8LL * i);
    v11 = *v10;
    if ( *v10 == -8192 )
      break;
    if ( v11 == -4096 )
      goto LABEL_8;
    if ( *(_QWORD *)(a2 + 8) == *(_QWORD *)(v11 + 8)
      && *(_BYTE *)(a2 + 56) == *(_BYTE *)(v11 + 96)
      && *(_BYTE *)(a2 + 57) == *(_BYTE *)(v11 + 97)
      && *(_DWORD *)(a2 + 60) == *(_DWORD *)(v11 + 100) )
    {
      v13 = *(_QWORD *)(a2 + 24);
      if ( *(_QWORD *)(v11 + 32) == v13 )
      {
        if ( !v13 )
          goto LABEL_31;
        v19 = v6;
        v22 = (__int64 *)(v6 + 8LL * i);
        v25 = v7;
        v28 = i;
        v14 = memcmp(*(const void **)(a2 + 16), *(const void **)(v11 + 24), v13);
        i = v28;
        v7 = v25;
        v10 = v22;
        v6 = v19;
        if ( !v14 )
        {
LABEL_31:
          v15 = *(_QWORD *)(a2 + 40);
          if ( *(_QWORD *)(v11 + 64) == v15 )
          {
            if ( !v15 )
              goto LABEL_20;
            v20 = v6;
            v23 = v10;
            v26 = v7;
            v29 = i;
            v16 = memcmp(*(const void **)(a2 + 32), *(const void **)(v11 + 56), v15);
            i = v29;
            v7 = v26;
            v10 = v23;
            v6 = v20;
            if ( !v16 )
            {
LABEL_20:
              v18 = v6;
              v21 = v10;
              v24 = v7;
              v27 = i;
              v30 = *(_QWORD *)(a2 + 48);
              v17 = sub_B3B7D0(v11);
              i = v27;
              v7 = v24;
              v10 = v21;
              v6 = v18;
              if ( v30 == v17 && *(_BYTE *)(a2 + 64) == *(_BYTE *)(v11 + 104) )
              {
                *a3 = v21;
                return 1;
              }
              v11 = *v21;
              break;
            }
          }
        }
      }
    }
LABEL_6:
    v12 = v7 + i;
    ++v7;
  }
  if ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    goto LABEL_6;
  }
LABEL_8:
  if ( !v8 )
    v8 = v10;
  *a3 = v8;
  return 0;
}
