// Function: sub_1A898D0
// Address: 0x1a898d0
//
__int64 __fastcall sub_1A898D0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 *v7; // r12
  __int64 *v8; // rax
  __int64 v9; // rsi
  __int64 *v10; // rbx
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 result; // rax
  __int64 *v14; // r8
  __int64 *v15; // r9
  __int64 *v16; // rax
  __int64 *v17; // rdi
  unsigned int v18; // r10d
  __int64 *v19; // rax
  __int64 *v20; // rcx

  v3 = *(__int64 **)(a1 + 8);
  v5 = *(__int64 **)(a1 + 16);
  if ( v5 == v3 )
    v6 = *(unsigned int *)(a1 + 28);
  else
    v6 = *(unsigned int *)(a1 + 24);
  v7 = &v5[v6];
  if ( v5 != v7 )
  {
    v8 = v5;
    while ( 1 )
    {
      v9 = *v8;
      v10 = v8;
      if ( (unsigned __int64)*v8 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v7 == ++v8 )
        goto LABEL_7;
    }
    if ( v8 != v7 )
    {
      v14 = *(__int64 **)(a2 + 16);
      v15 = *(__int64 **)(a2 + 8);
      if ( v14 == v15 )
        goto LABEL_23;
LABEL_16:
      sub_16CCBA0(a2, v9);
      v14 = *(__int64 **)(a2 + 16);
      v15 = *(__int64 **)(a2 + 8);
LABEL_17:
      while ( 1 )
      {
        v16 = v10 + 1;
        if ( v10 + 1 == v7 )
          break;
        while ( 1 )
        {
          v9 = *v16;
          v10 = v16;
          if ( (unsigned __int64)*v16 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v7 == ++v16 )
            goto LABEL_20;
        }
        if ( v16 == v7 )
          break;
        if ( v14 != v15 )
          goto LABEL_16;
LABEL_23:
        v17 = &v14[*(unsigned int *)(a2 + 28)];
        v18 = *(_DWORD *)(a2 + 28);
        if ( v17 == v14 )
        {
LABEL_31:
          if ( v18 >= *(_DWORD *)(a2 + 24) )
            goto LABEL_16;
          *(_DWORD *)(a2 + 28) = v18 + 1;
          *v17 = v9;
          v15 = *(__int64 **)(a2 + 8);
          ++*(_QWORD *)a2;
          v14 = *(__int64 **)(a2 + 16);
        }
        else
        {
          v19 = v14;
          v20 = 0;
          while ( *v19 != v9 )
          {
            if ( *v19 == -2 )
              v20 = v19;
            if ( v17 == ++v19 )
            {
              if ( !v20 )
                goto LABEL_31;
              *v20 = v9;
              v14 = *(__int64 **)(a2 + 16);
              --*(_DWORD *)(a2 + 32);
              v15 = *(__int64 **)(a2 + 8);
              ++*(_QWORD *)a2;
              goto LABEL_17;
            }
          }
        }
      }
LABEL_20:
      v5 = *(__int64 **)(a1 + 16);
      v3 = *(__int64 **)(a1 + 8);
    }
  }
LABEL_7:
  ++*(_QWORD *)a1;
  if ( v5 != v3 )
  {
    v11 = 4 * (*(_DWORD *)(a1 + 28) - *(_DWORD *)(a1 + 32));
    v12 = *(unsigned int *)(a1 + 24);
    if ( v11 < 0x20 )
      v11 = 32;
    if ( (unsigned int)v12 > v11 )
    {
      sub_16CC920(a1);
      goto LABEL_13;
    }
    memset(v5, -1, 8 * v12);
  }
  *(_QWORD *)(a1 + 28) = 0;
LABEL_13:
  result = *(unsigned __int8 *)(a1 + 104);
  *(_BYTE *)(a2 + 104) |= result;
  return result;
}
