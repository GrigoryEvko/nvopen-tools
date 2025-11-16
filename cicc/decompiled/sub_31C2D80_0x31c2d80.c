// Function: sub_31C2D80
// Address: 0x31c2d80
//
__int64 __fastcall sub_31C2D80(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r9
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r12
  __int64 v14; // rcx
  unsigned int v15; // edi
  _QWORD *v16; // rax
  _QWORD *v17; // rcx
  unsigned int v18; // r13d
  __int64 i; // rcx
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 *v24; // rax
  __int64 v25; // rdx
  int v27; // edx
  int v28; // ecx
  unsigned __int64 v29; // rdx
  int v30; // ecx
  unsigned int v31; // eax
  int v32; // ecx
  __int64 v33; // r8
  unsigned __int64 v34; // r14
  int v35; // r10d
  _QWORD v36[7]; // [rsp-38h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 56);
  v4 = *(unsigned int *)(a1 + 72);
  if ( !(_DWORD)v4 )
    return 0;
  v5 = (unsigned int)(v4 - 1);
  v6 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v3 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v27 = 1;
    while ( v8 != -4096 )
    {
      v35 = v27 + 1;
      v6 = v5 & (v27 + v6);
      v7 = (__int64 *)(v3 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v27 = v35;
    }
    return 0;
  }
LABEL_3:
  if ( v7 == (__int64 *)(v3 + 16 * v4) )
    return 0;
  v9 = v7[1];
  v10 = *(_QWORD *)(v9 + 8);
  v11 = 8LL * *(unsigned int *)(v9 + 16);
  v12 = v10 + v11;
  v13 = v11 >> 3;
  v14 = v11 >> 5;
  v15 = v13;
  if ( v14 )
  {
    v16 = *(_QWORD **)(v9 + 8);
    v17 = (_QWORD *)(v10 + 32 * v14);
    while ( a2 != *v16 )
    {
      if ( a2 == v16[1] )
      {
        v15 = ((__int64)v16 - v10 + 8) >> 3;
        v13 = v15;
        goto LABEL_12;
      }
      if ( a2 == v16[2] )
      {
        v15 = ((__int64)v16 - v10 + 16) >> 3;
        v13 = v15;
        goto LABEL_12;
      }
      if ( a2 == v16[3] )
      {
        v15 = ((__int64)v16 - v10 + 24) >> 3;
        v13 = v15;
        goto LABEL_12;
      }
      v16 += 4;
      if ( v17 == v16 )
      {
        v33 = (v12 - (__int64)v16) >> 3;
        goto LABEL_37;
      }
    }
    goto LABEL_11;
  }
  v33 = v13;
  v16 = *(_QWORD **)(v9 + 8);
LABEL_37:
  if ( v33 == 2 )
  {
LABEL_49:
    if ( a2 != *v16 )
    {
      ++v16;
LABEL_40:
      if ( a2 != *v16 )
        goto LABEL_12;
      goto LABEL_11;
    }
    goto LABEL_11;
  }
  if ( v33 != 3 )
  {
    if ( v33 != 1 )
      goto LABEL_12;
    goto LABEL_40;
  }
  if ( a2 != *v16 )
  {
    ++v16;
    goto LABEL_49;
  }
LABEL_11:
  v15 = ((__int64)v16 - v10) >> 3;
  v13 = v15;
LABEL_12:
  v18 = v15 + 1;
  if ( v15 + 1 >= *(_DWORD *)(v9 + 136) )
  {
    v28 = *(_DWORD *)(v9 + 136) & 0x3F;
    if ( v28 )
      *(_QWORD *)(*(_QWORD *)(v9 + 72) + 8LL * *(unsigned int *)(v9 + 80) - 8) &= ~(-1LL << v28);
    v29 = *(unsigned int *)(v9 + 80);
    *(_DWORD *)(v9 + 136) = v18;
    LOBYTE(v30) = v15 + 1;
    v31 = (v15 + 64) >> 6;
    if ( v31 != v29 )
    {
      if ( v31 >= v29 )
      {
        v34 = v31 - v29;
        if ( v31 > (unsigned __int64)*(unsigned int *)(v9 + 84) )
        {
          sub_C8D5F0(v9 + 72, (const void *)(v9 + 88), v31, 8u, v31, v5);
          v29 = *(unsigned int *)(v9 + 80);
        }
        if ( 8 * v34 )
        {
          memset((void *)(*(_QWORD *)(v9 + 72) + 8 * v29), 0, 8 * v34);
          LODWORD(v29) = *(_DWORD *)(v9 + 80);
        }
        v30 = *(_DWORD *)(v9 + 136);
        *(_DWORD *)(v9 + 80) = v34 + v29;
      }
      else
      {
        *(_DWORD *)(v9 + 80) = v31;
      }
    }
    v32 = v30 & 0x3F;
    if ( v32 )
      *(_QWORD *)(*(_QWORD *)(v9 + 72) + 8LL * *(unsigned int *)(v9 + 80) - 8) &= ~(-1LL << v32);
  }
  for ( i = v13; i != v18; ++*(_DWORD *)(v9 + 144) )
  {
    v20 = 1LL << i;
    v21 = (unsigned int)i++ >> 6;
    *(_QWORD *)(*(_QWORD *)(v9 + 72) + 8 * v21) |= v20;
  }
  v22 = *(_QWORD *)(*(_QWORD *)(v9 + 8) + 8 * v13);
  v23 = sub_B43CA0(*(_QWORD *)(v22 + 16)) + 312;
  if ( sub_318B630(v22) && (*(_DWORD *)(v22 + 8) != 37 || sub_318B6C0(v22)) )
  {
    if ( sub_318B670(v22) )
    {
      v22 = sub_318B680(v22);
    }
    else if ( *(_DWORD *)(v22 + 8) == 37 )
    {
      v22 = sub_318B6C0(v22);
    }
  }
  v24 = sub_318EB80(v22);
  v36[0] = sub_9208B0(v23, *v24);
  v36[1] = v25;
  *(_DWORD *)(v9 + 148) -= sub_CA1930(v36);
  return 1;
}
