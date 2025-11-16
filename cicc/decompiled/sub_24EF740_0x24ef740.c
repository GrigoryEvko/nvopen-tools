// Function: sub_24EF740
// Address: 0x24ef740
//
__int64 __fastcall sub_24EF740(__int64 a1, __int64 *a2)
{
  __int64 v4; // rsi
  char v5; // di
  int v6; // edi
  __int64 v7; // r9
  __int64 v8; // r8
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r10
  __int64 v12; // rcx
  __int64 result; // rax
  __int64 v14; // r12
  unsigned int v15; // eax
  int v16; // edx
  unsigned int v17; // edi
  __int64 *v18; // rax
  int v19; // r13d
  __int64 *v20; // r11
  unsigned int v21; // esi
  __int64 *v22; // [rsp+8h] [rbp-38h] BYREF
  __int64 v23; // [rsp+10h] [rbp-30h] BYREF
  __int64 v24; // [rsp+18h] [rbp-28h]

  v4 = *a2;
  v24 = *(unsigned int *)(a1 + 88);
  v5 = *(_BYTE *)(a1 + 8);
  v23 = v4;
  v6 = v5 & 1;
  if ( v6 )
  {
    v7 = a1 + 16;
    v8 = 3;
  }
  else
  {
    v8 = *(unsigned int *)(a1 + 24);
    v7 = *(_QWORD *)(a1 + 16);
    if ( !(_DWORD)v8 )
    {
      v15 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v22 = 0;
      v16 = (v15 >> 1) + 1;
      goto LABEL_13;
    }
    v8 = (unsigned int)(v8 - 1);
  }
  v9 = v8 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v4 != *v10 )
  {
    v19 = 1;
    v20 = 0;
    while ( v11 != -4096 )
    {
      if ( !v20 && v11 == -8192 )
        v20 = v10;
      v9 = v8 & (v19 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( v4 == *v10 )
        goto LABEL_4;
      ++v19;
    }
    v15 = *(_DWORD *)(a1 + 8);
    if ( !v20 )
      v20 = v10;
    ++*(_QWORD *)a1;
    v22 = v20;
    v16 = (v15 >> 1) + 1;
    if ( (_BYTE)v6 )
    {
      v17 = 12;
      v8 = 4;
LABEL_14:
      if ( v17 <= 4 * v16 )
      {
        v21 = 2 * v8;
      }
      else
      {
        if ( (int)v8 - *(_DWORD *)(a1 + 12) - v16 > (unsigned int)v8 >> 3 )
        {
LABEL_16:
          *(_DWORD *)(a1 + 8) = (2 * (v15 >> 1) + 2) | v15 & 1;
          v18 = v22;
          if ( *v22 != -4096 )
            --*(_DWORD *)(a1 + 12);
          *v18 = v4;
          v18[1] = v24;
          goto LABEL_9;
        }
        v21 = v8;
      }
      sub_227FE50(a1, v21);
      sub_227C450(a1, &v23, &v22);
      v4 = v23;
      v15 = *(_DWORD *)(a1 + 8);
      goto LABEL_16;
    }
    v8 = *(unsigned int *)(a1 + 24);
LABEL_13:
    v17 = 3 * v8;
    goto LABEL_14;
  }
LABEL_4:
  v12 = v10[1];
  result = v24 - 1;
  if ( v12 == v24 - 1 )
    return result;
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v12) = 0;
  v10[1] = *(unsigned int *)(a1 + 88);
LABEL_9:
  result = *(unsigned int *)(a1 + 88);
  v14 = *a2;
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
  {
    sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), result + 1, 8u, v8, v7);
    result = *(unsigned int *)(a1 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * result) = v14;
  ++*(_DWORD *)(a1 + 88);
  return result;
}
