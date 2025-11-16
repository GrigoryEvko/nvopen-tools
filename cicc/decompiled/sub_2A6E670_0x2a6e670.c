// Function: sub_2A6E670
// Address: 0x2a6e670
//
__int64 __fastcall sub_2A6E670(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // r8
  __int64 v6; // rdx
  int v7; // r10d
  __int64 v8; // r9
  unsigned int i; // eax
  __int64 *v10; // rcx
  __int64 v11; // r11
  unsigned int v12; // eax
  __int64 result; // rax
  int v14; // eax
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rbx
  __int64 v20; // rbx
  int v21; // esi
  __int64 *v22; // [rsp+8h] [rbp-38h] BYREF
  __int64 v23; // [rsp+10h] [rbp-30h] BYREF
  __int64 v24; // [rsp+18h] [rbp-28h]

  v3 = a1 + 2504;
  v5 = *(unsigned int *)(a1 + 2528);
  v23 = a2;
  v24 = a3;
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(a1 + 2504);
    v22 = 0;
LABEL_34:
    v21 = 2 * v5;
LABEL_35:
    sub_2884B10(v3, v21);
    sub_2A68180(v3, &v23, &v22);
    a2 = v23;
    v10 = v22;
    v15 = (unsigned int)(*(_DWORD *)(a1 + 2520) + 1);
    goto LABEL_18;
  }
  v6 = *(_QWORD *)(a1 + 2512);
  v7 = 1;
  v8 = 0;
  for ( i = (v5 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v5 - 1) & v12 )
  {
    v10 = (__int64 *)(v6 + 16LL * i);
    v11 = *v10;
    if ( a2 == *v10 && a3 == v10[1] )
      return 0;
    if ( v11 == -4096 )
      break;
    if ( v11 == -8192 && v10[1] == -8192 && !v8 )
      v8 = v6 + 16LL * i;
LABEL_9:
    v12 = v7 + i;
    ++v7;
  }
  if ( v10[1] != -4096 )
    goto LABEL_9;
  v14 = *(_DWORD *)(a1 + 2520);
  if ( v8 )
    v10 = (__int64 *)v8;
  ++*(_QWORD *)(a1 + 2504);
  v15 = (unsigned int)(v14 + 1);
  v22 = v10;
  if ( 4 * (int)v15 >= (unsigned int)(3 * v5) )
    goto LABEL_34;
  if ( (int)v5 - *(_DWORD *)(a1 + 2524) - (int)v15 <= (unsigned int)v5 >> 3 )
  {
    v21 = v5;
    goto LABEL_35;
  }
LABEL_18:
  *(_DWORD *)(a1 + 2520) = v15;
  if ( *v10 != -4096 || v10[1] != -4096 )
    --*(_DWORD *)(a1 + 2524);
  *v10 = a2;
  v10[1] = v24;
  result = sub_2A62EB0(a1, a3, (__int64 *)v15, (__int64)v10, v5, v8);
  if ( !(_BYTE)result )
  {
    v16 = sub_AA5930(a3);
    v18 = v17;
    v19 = v16;
    if ( v16 != v17 )
    {
      while ( 1 )
      {
        sub_2A6ACC0(a1, v19);
        if ( !v19 )
          goto LABEL_29;
        v20 = *(_QWORD *)(v19 + 32);
        if ( !v20 )
          BUG();
        if ( *(_BYTE *)(v20 - 24) != 84 )
          break;
        v19 = v20 - 24;
        if ( v18 == v19 )
          return 1;
      }
      if ( v18 )
      {
        sub_2A6ACC0(a1, 0);
LABEL_29:
        BUG();
      }
    }
    return 1;
  }
  return result;
}
