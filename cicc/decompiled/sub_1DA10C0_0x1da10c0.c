// Function: sub_1DA10C0
// Address: 0x1da10c0
//
__int64 __fastcall sub_1DA10C0(__int64 **a1, __int64 *a2, __int64 *a3)
{
  char v5; // cl
  __int64 **v6; // rdx
  int v7; // r8d
  int v8; // r10d
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // r9
  unsigned int i; // eax
  __int64 **v12; // r13
  unsigned int v13; // eax
  __int64 v14; // r8
  __int64 v15; // r13
  __int64 result; // rax
  __int64 *v17; // rdi
  __int64 *v18; // rdx
  __int64 *v19; // r12
  unsigned int v20; // ecx
  unsigned int v21; // esi
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int64 *v24; // rax

  v5 = (_BYTE)a1[5] & 1;
  if ( v5 )
  {
    v6 = a1 + 6;
    v7 = 7;
  }
  else
  {
    v14 = *((unsigned int *)a1 + 14);
    v6 = (__int64 **)a1[6];
    if ( !(_DWORD)v14 )
    {
LABEL_10:
      v15 = 3 * v14;
LABEL_11:
      v12 = &v6[v15];
      goto LABEL_12;
    }
    v7 = v14 - 1;
  }
  v8 = 1;
  v9 = (((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
        | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
     ^ ((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
       | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
      - 1
      - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32));
  v10 = ((9 * (((v9 - 1 - (v9 << 13)) >> 8) ^ (v9 - 1 - (v9 << 13)))) >> 15)
      ^ (9 * (((v9 - 1 - (v9 << 13)) >> 8) ^ (v9 - 1 - (v9 << 13))));
  for ( i = v7 & (((v10 - 1 - (v10 << 27)) >> 31) ^ (v10 - 1 - ((_DWORD)v10 << 27))); ; i = v7 & v13 )
  {
    v12 = &v6[3 * i];
    if ( *v12 == a2 && v12[1] == a3 )
      break;
    if ( *v12 == (__int64 *)-8LL && v12[1] == (__int64 *)-8LL )
    {
      if ( !v5 )
      {
        v14 = *((unsigned int *)a1 + 14);
        goto LABEL_10;
      }
      v15 = 24;
      goto LABEL_11;
    }
    v13 = v8 + i;
    ++v8;
  }
LABEL_12:
  result = 192;
  if ( !v5 )
    result = 24LL * *((unsigned int *)a1 + 14);
  if ( v12 != (__int64 **)((char *)v6 + result) )
  {
    v17 = a1[1];
    v18 = (__int64 *)(a1 + 1);
    if ( v17 != (__int64 *)(a1 + 1) )
    {
      v19 = *a1;
      v20 = *((_DWORD *)v12 + 4);
      if ( v18 == *a1 )
      {
        v19 = (__int64 *)v19[1];
        v22 = v20 >> 7;
        *a1 = v19;
        v21 = *((_DWORD *)v19 + 4);
        if ( v20 >> 7 == v21 )
        {
          if ( v18 == v19 )
            goto LABEL_24;
          goto LABEL_30;
        }
      }
      else
      {
        v21 = *((_DWORD *)v19 + 4);
        v22 = v20 >> 7;
        if ( v21 == v20 >> 7 )
          goto LABEL_30;
      }
      if ( v21 <= v22 )
      {
        if ( v18 == v19 )
        {
LABEL_42:
          *a1 = v19;
          goto LABEL_24;
        }
        while ( v21 < v22 )
        {
          v19 = (__int64 *)*v19;
          if ( v18 == v19 )
            goto LABEL_42;
          v21 = *((_DWORD *)v19 + 4);
        }
      }
      else
      {
        if ( v17 == v19 )
        {
          *a1 = v19;
LABEL_23:
          if ( *((_DWORD *)v19 + 4) != v22 )
            goto LABEL_24;
LABEL_30:
          v19[((v20 >> 6) & 1) + 3] &= ~(1LL << v20);
          if ( !v19[3] && !v19[4] )
          {
            v24 = (__int64 *)**a1;
            a1[3] = (__int64 *)((char *)a1[3] - 1);
            *a1 = v24;
            sub_2208CA0(v19);
            j_j___libc_free_0(v19, 40);
          }
          goto LABEL_24;
        }
        do
          v19 = (__int64 *)v19[1];
        while ( v17 != v19 && *((_DWORD *)v19 + 4) > v22 );
      }
      *a1 = v19;
      if ( v18 != v19 )
        goto LABEL_23;
    }
LABEL_24:
    *v12 = (__int64 *)-16LL;
    v12[1] = (__int64 *)-16LL;
    v23 = *((_DWORD *)a1 + 10);
    ++*((_DWORD *)a1 + 11);
    result = (2 * (v23 >> 1) - 2) | v23 & 1;
    *((_DWORD *)a1 + 10) = result;
  }
  return result;
}
