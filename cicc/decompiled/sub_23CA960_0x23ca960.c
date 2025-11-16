// Function: sub_23CA960
// Address: 0x23ca960
//
void __fastcall sub_23CA960(__int64 a1)
{
  unsigned int v1; // eax
  __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r9
  unsigned int v5; // r13d
  unsigned __int64 v6; // r8
  __int64 v7; // r9
  unsigned int i; // r12d
  int v9; // eax
  __int64 v10; // r9
  __int64 v11; // rdx
  unsigned __int64 v12; // r8
  __int64 v13; // rax
  int *v14; // rbx
  int *v15; // r13
  __int64 v16; // rax
  int *v17; // r15
  int v18; // r13d
  _DWORD *v19; // rax
  _DWORD *v20; // r15
  _DWORD *v21; // r12
  __int64 v22; // rax
  __int64 v23; // rdx
  _DWORD *v24; // rax
  _DWORD *v25; // r15
  _DWORD *v26; // r12
  int v27; // eax
  __int64 v28; // rdx
  int v29; // [rsp+4h] [rbp-9Ch]
  __int64 v30; // [rsp+28h] [rbp-78h]
  int v31; // [rsp+28h] [rbp-78h]
  int *v32; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-68h]
  unsigned int v34; // [rsp+3Ch] [rbp-64h]
  _BYTE v35[96]; // [rsp+40h] [rbp-60h] BYREF

  v32 = (int *)v35;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = 0;
  v34 = 12;
  while ( 1 )
  {
    v1 = *(_DWORD *)(a1 + 88);
    if ( !v1 )
      break;
    v2 = *(_QWORD *)(a1 + 80);
    v33 = 0;
    v3 = *(_QWORD *)(v2 + 8LL * v1 - 8);
    *(_DWORD *)(a1 + 88) = v1 - 1;
    v5 = sub_23CC710(v3);
    if ( *(_DWORD *)(v3 + 56) )
    {
      v19 = *(_DWORD **)(v3 + 48);
      v20 = &v19[4 * *(unsigned int *)(v3 + 64)];
      if ( v19 != v20 )
      {
        while ( 1 )
        {
          v21 = v19;
          if ( *v19 <= 0xFFFFFFFD )
            break;
          v19 += 4;
          if ( v20 == v19 )
            goto LABEL_4;
        }
        while ( v20 != v21 )
        {
          v22 = *((_QWORD *)v21 + 1);
          if ( *(_DWORD *)(v22 + 8) == 1 )
          {
            v23 = *(unsigned int *)(a1 + 88);
            if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
            {
              v30 = *((_QWORD *)v21 + 1);
              sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v23 + 1, 8u, v23 + 1, v4);
              v23 = *(unsigned int *)(a1 + 88);
              v22 = v30;
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v23) = v22;
            ++*(_DWORD *)(a1 + 88);
          }
          v21 += 4;
          if ( v21 == v20 )
            break;
          while ( *v21 > 0xFFFFFFFD )
          {
            v21 += 4;
            if ( v20 == v21 )
              goto LABEL_4;
          }
        }
      }
    }
LABEL_4:
    if ( *(_DWORD *)(a1 + 144) <= v5 && !(unsigned __int8)sub_23CC720(v3) )
    {
      if ( *(_BYTE *)(a1 + 160) )
      {
        for ( i = sub_23CC780(v3); (unsigned int)sub_23CC790(v3) >= i; ++i )
        {
          v9 = sub_23CC760(*(_QWORD *)(**(_QWORD **)(a1 + 152) + 8LL * i));
          v11 = v33;
          v12 = v33 + 1LL;
          if ( v12 > v34 )
          {
            v31 = v9;
            sub_C8D5F0((__int64)&v32, v35, v33 + 1LL, 4u, v12, v10);
            v11 = v33;
            v9 = v31;
          }
          v32[v11] = v9;
          ++v33;
        }
      }
      else if ( *(_DWORD *)(v3 + 56) )
      {
        v24 = *(_DWORD **)(v3 + 48);
        v6 = 16LL * *(unsigned int *)(v3 + 64);
        v25 = (_DWORD *)((char *)v24 + v6);
        if ( v24 != (_DWORD *)((char *)v24 + v6) )
        {
          while ( 1 )
          {
            v26 = v24;
            if ( *v24 <= 0xFFFFFFFD )
              break;
            v24 += 4;
            if ( v25 == v24 )
              goto LABEL_12;
          }
          if ( v25 != v24 )
          {
            do
            {
              if ( !*(_DWORD *)(*((_QWORD *)v26 + 1) + 8LL) )
              {
                v27 = ((__int64 (*)(void))sub_23CC760)();
                v28 = v33;
                v6 = v33 + 1LL;
                if ( v6 > v34 )
                {
                  v29 = v27;
                  sub_C8D5F0((__int64)&v32, v35, v33 + 1LL, 4u, v6, v7);
                  v28 = v33;
                  v27 = v29;
                }
                v32[v28] = v27;
                ++v33;
              }
              v26 += 4;
              if ( v26 == v25 )
                break;
              while ( *v26 > 0xFFFFFFFD )
              {
                v26 += 4;
                if ( v25 == v26 )
                  goto LABEL_12;
              }
            }
            while ( v26 != v25 );
          }
        }
      }
LABEL_12:
      v13 = v33;
      if ( v33 > 1uLL )
      {
        *(_QWORD *)a1 = v3;
        v14 = v32;
        *(_DWORD *)(a1 + 8) = v5;
        v15 = &v14[v13];
        v16 = *(unsigned int *)(a1 + 24);
        v17 = v15;
        do
        {
          v18 = *v14;
          if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
          {
            sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v16 + 1, 4u, v6, v7);
            v16 = *(unsigned int *)(a1 + 24);
          }
          ++v14;
          *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v16) = v18;
          v16 = (unsigned int)(*(_DWORD *)(a1 + 24) + 1);
          *(_DWORD *)(a1 + 24) = v16;
        }
        while ( v17 != v14 );
        break;
      }
    }
  }
  if ( v32 != (int *)v35 )
    _libc_free((unsigned __int64)v32);
}
