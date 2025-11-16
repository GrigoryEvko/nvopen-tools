// Function: sub_1E89C70
// Address: 0x1e89c70
//
__int64 __fastcall sub_1E89C70(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  int v3; // r9d
  unsigned int v4; // eax
  __int64 i; // r8
  __int64 v6; // rcx
  unsigned int *v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rdx
  unsigned int *v10; // r12
  unsigned int v11; // eax
  int v12; // edx
  __int64 result; // rax
  __int64 v14; // rdx
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v16[0] = a2;
  v2 = sub_1E855C0(a1 + 528, v16);
  sub_1E898E0((__int64)(v2 + 6), a1 + 392);
  sub_1E89640(a1 + 200, a1 + 392);
  v4 = *(_DWORD *)(a1 + 480);
  *(_DWORD *)(a1 + 400) = 0;
  for ( i = a1 + 312; v4; v4 = *(_DWORD *)(a1 + 480) )
  {
    while ( 1 )
    {
      v6 = v4;
      v7 = *(unsigned int **)(a1 + 208);
      --v4;
      v8 = *(_QWORD *)(*(_QWORD *)(a1 + 472) + 8 * v6 - 8);
      v9 = *(unsigned int *)(a1 + 224);
      *(_DWORD *)(a1 + 480) = v4;
      v10 = &v7[v9];
      if ( *(_DWORD *)(a1 + 216) )
      {
        if ( v7 != v10 )
        {
          while ( *v7 > 0xFFFFFFFD )
          {
            if ( ++v7 == v10 )
              goto LABEL_3;
          }
          if ( v7 != v10 )
            break;
        }
      }
LABEL_3:
      if ( !v4 )
        goto LABEL_16;
    }
LABEL_10:
    v11 = *v7;
    if ( (int)*v7 > 0 )
    {
      v12 = *(_DWORD *)(v8 + 4LL * (v11 >> 5));
      if ( !_bittest(&v12, v11) )
      {
        v14 = *(unsigned int *)(a1 + 320);
        if ( (unsigned int)v14 >= *(_DWORD *)(a1 + 324) )
        {
          v15 = i;
          sub_16CD150(i, (const void *)(a1 + 328), 0, 4, i, v3);
          v14 = *(unsigned int *)(a1 + 320);
          v11 = *v7;
          i = v15;
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 312) + 4 * v14) = v11;
        ++*(_DWORD *)(a1 + 320);
      }
    }
    while ( ++v7 != v10 )
    {
      if ( *v7 <= 0xFFFFFFFD )
      {
        if ( v7 != v10 )
          goto LABEL_10;
        break;
      }
    }
  }
LABEL_16:
  sub_1E89640(a1 + 200, i);
  *(_DWORD *)(a1 + 320) = 0;
  result = sub_1E898E0(a1 + 200, a1 + 232);
  *(_DWORD *)(a1 + 240) = 0;
  return result;
}
