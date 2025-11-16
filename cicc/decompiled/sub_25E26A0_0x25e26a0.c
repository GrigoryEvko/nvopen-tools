// Function: sub_25E26A0
// Address: 0x25e26a0
//
__int64 __fastcall sub_25E26A0(__int64 a1, __int64 a2)
{
  char v3; // cl
  int v4; // ecx
  __int64 v5; // r8
  int v6; // esi
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 result; // rax
  unsigned int v11; // esi
  unsigned int v12; // eax
  int v13; // edx
  unsigned int v14; // edi
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // r14
  __int64 v22; // rdi
  int v23; // r11d
  __int64 v24; // r10
  __int64 v25; // [rsp+8h] [rbp-48h] BYREF
  __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_BYTE *)(a2 + 8);
  v25 = a1;
  v4 = v3 & 1;
  if ( v4 )
  {
    v5 = a2 + 16;
    v6 = 7;
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 16);
    v11 = *(_DWORD *)(a2 + 24);
    if ( !v11 )
    {
      v12 = *(_DWORD *)(a2 + 8);
      ++*(_QWORD *)a2;
      v26[0] = 0;
      v13 = (v12 >> 1) + 1;
LABEL_8:
      v14 = 3 * v11;
      goto LABEL_9;
    }
    v6 = v11 - 1;
  }
  v7 = v6 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
  v8 = v5 + 16LL * v7;
  v9 = *(_QWORD *)v8;
  if ( v25 == *(_QWORD *)v8 )
    return *(unsigned __int8 *)(v8 + 8);
  v23 = 1;
  v24 = 0;
  while ( v9 != -4096 )
  {
    if ( v9 == -8192 && !v24 )
      v24 = v8;
    v7 = v6 & (v23 + v7);
    v8 = v5 + 16LL * v7;
    v9 = *(_QWORD *)v8;
    if ( v25 == *(_QWORD *)v8 )
      return *(unsigned __int8 *)(v8 + 8);
    ++v23;
  }
  if ( !v24 )
    v24 = v8;
  v12 = *(_DWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  v26[0] = v24;
  v13 = (v12 >> 1) + 1;
  if ( !(_BYTE)v4 )
  {
    v11 = *(_DWORD *)(a2 + 24);
    goto LABEL_8;
  }
  v14 = 24;
  v11 = 8;
LABEL_9:
  if ( v14 <= 4 * v13 )
  {
    v11 *= 2;
  }
  else if ( v11 - *(_DWORD *)(a2 + 12) - v13 > v11 >> 3 )
  {
    goto LABEL_11;
  }
  sub_25E2260(a2, v11);
  sub_25E0BC0(a2, &v25, v26);
  v12 = *(_DWORD *)(a2 + 8);
LABEL_11:
  v15 = v26[0];
  *(_DWORD *)(a2 + 8) = (2 * (v12 >> 1) + 2) | v12 & 1;
  if ( *(_QWORD *)v15 != -4096 )
    --*(_DWORD *)(a2 + 12);
  v16 = v25;
  *(_BYTE *)(v15 + 8) = 0;
  *(_QWORD *)v15 = v16;
  v17 = v25;
  if ( (((*(_WORD *)(v25 + 2) >> 4) & 0x3FF) == 0 || ((*(_WORD *)(v25 + 2) >> 4) & 0x3FF) == 0x46)
    && !(*(_DWORD *)(*(_QWORD *)(v25 + 24) + 8LL) >> 8) )
  {
    v18 = *(_QWORD *)(v25 + 16);
    if ( v18 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v19 = *(_QWORD *)(v18 + 24);
          if ( *(_BYTE *)v19 == 85 )
            break;
          v18 = *(_QWORD *)(v18 + 8);
          if ( !v18 )
            goto LABEL_24;
        }
        if ( (*(_WORD *)(v19 + 2) & 3) == 2 )
          break;
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          goto LABEL_24;
      }
    }
    else
    {
LABEL_24:
      v20 = *(_QWORD *)(v25 + 80);
      v21 = v25 + 72;
      if ( v20 == v25 + 72 )
      {
LABEL_37:
        result = (unsigned int)sub_B2DDD0(v17, 0, 0, 1, 0, 0, 0) ^ 1;
        goto LABEL_16;
      }
      while ( 1 )
      {
        v22 = v20 - 24;
        if ( !v20 )
          v22 = 0;
        if ( sub_AA4E50(v22) )
          break;
        v20 = *(_QWORD *)(v20 + 8);
        if ( v21 == v20 )
          goto LABEL_37;
      }
    }
  }
  result = 0;
LABEL_16:
  *(_BYTE *)(v15 + 8) = result;
  return result;
}
