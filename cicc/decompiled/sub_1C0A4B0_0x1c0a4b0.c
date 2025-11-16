// Function: sub_1C0A4B0
// Address: 0x1c0a4b0
//
__int64 __fastcall sub_1C0A4B0(int *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r9
  __int64 v9; // r12
  int v11; // edx
  int v12; // eax
  unsigned int v13; // r13d
  __int64 v14; // rax
  size_t v15; // rdx
  void *v16; // r14
  __int64 *v17; // rax
  __int64 *v18; // rsi
  char v19; // r8
  __int64 *v20; // rax
  int v21; // edi
  unsigned int v22; // esi
  int v23; // edx
  __int64 v24; // rdx
  __int64 v25; // rdx
  int v26; // r10d
  __int64 v27; // rax
  size_t n; // [rsp+0h] [rbp-50h]
  __int64 v29[2]; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v30[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = (unsigned int)a1[8];
  v29[0] = a2;
  if ( !(_DWORD)v3 )
    goto LABEL_8;
  v5 = *((_QWORD *)a1 + 2);
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v11 = 1;
    while ( v8 != -8 )
    {
      v26 = v11 + 1;
      v6 = (v3 - 1) & (v11 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v11 = v26;
    }
LABEL_8:
    v9 = sub_22077B0(24);
    if ( v9 )
    {
      v12 = *a1;
      *(_QWORD *)v9 = 0;
      *(_QWORD *)(v9 + 8) = 0;
      *(_DWORD *)(v9 + 16) = v12;
      v13 = (unsigned int)(v12 + 63) >> 6;
      v14 = malloc(8LL * v13);
      v15 = 8LL * v13;
      v16 = (void *)v14;
      if ( !v14 )
      {
        if ( 8LL * v13 || (v27 = malloc(1u), v15 = 0, !v27) )
        {
          n = v15;
          sub_16BD1C0("Allocation failed", 1u);
          v15 = n;
        }
        else
        {
          v16 = (void *)v27;
        }
      }
      *(_QWORD *)v9 = v16;
      *(_QWORD *)(v9 + 8) = v13;
      if ( v13 )
        memset(v16, 0, v15);
    }
    v17 = *(__int64 **)(v29[0] + 8);
    v18 = &v17[*(unsigned int *)(v29[0] + 24)];
    if ( *(_DWORD *)(v29[0] + 16) && v17 != v18 )
    {
      while ( *v17 == -16 || *v17 == -8 )
      {
        if ( v18 == ++v17 )
          goto LABEL_13;
      }
      while ( v18 != v17 )
      {
        v25 = *v17++;
        *(_QWORD *)(*(_QWORD *)v9 + 8LL * (*(_DWORD *)(v25 + 16) >> 6)) |= 1LL << *(_DWORD *)(v25 + 16);
        if ( v17 == v18 )
          break;
        while ( *v17 == -16 || *v17 == -8 )
        {
          if ( v18 == ++v17 )
            goto LABEL_13;
        }
      }
    }
LABEL_13:
    v19 = sub_1C098B0((__int64)(a1 + 2), v29, v30);
    v20 = (__int64 *)v30[0];
    if ( v19 )
      goto LABEL_19;
    v21 = a1[6];
    v22 = a1[8];
    ++*((_QWORD *)a1 + 1);
    v23 = v21 + 1;
    if ( 4 * (v21 + 1) >= 3 * v22 )
    {
      v22 *= 2;
    }
    else if ( v22 - a1[7] - v23 > v22 >> 3 )
    {
LABEL_16:
      a1[6] = v23;
      if ( *v20 != -8 )
        --a1[7];
      v24 = v29[0];
      v20[1] = 0;
      *v20 = v24;
LABEL_19:
      v20[1] = v9;
      return v9;
    }
    sub_1C0A2F0((__int64)(a1 + 2), v22);
    sub_1C098B0((__int64)(a1 + 2), v29, v30);
    v20 = (__int64 *)v30[0];
    v23 = a1[6] + 1;
    goto LABEL_16;
  }
LABEL_3:
  if ( v7 == (__int64 *)(v5 + 16 * v3) )
    goto LABEL_8;
  return v7[1];
}
