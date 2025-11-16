// Function: sub_BCCA50
// Address: 0xbcca50
//
__int64 __fastcall sub_BCCA50(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v3; // r12
  int v4; // r14d
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  _QWORD *v8; // rdi
  unsigned int v9; // r8d
  _QWORD *v10; // r15
  __int64 v11; // rax
  _QWORD *v12; // r9
  int v13; // r13d
  int v14; // r15d
  unsigned int v15; // r12d
  size_t v16; // rdx
  const void *v17; // rcx
  __int64 v18; // r14
  int v19; // eax
  size_t v20; // r14
  int v21; // eax
  size_t v22; // rdx
  int v23; // eax
  void *s2; // [rsp+8h] [rbp-78h]
  _QWORD *v25; // [rsp+10h] [rbp-70h]
  const void *v26; // [rsp+18h] [rbp-68h]
  _QWORD *v27; // [rsp+18h] [rbp-68h]
  unsigned int v28; // [rsp+20h] [rbp-60h]
  _QWORD *v29; // [rsp+20h] [rbp-60h]
  __int64 v30; // [rsp+28h] [rbp-58h]
  _QWORD *v31; // [rsp+30h] [rbp-50h]
  _QWORD *v32; // [rsp+38h] [rbp-48h]
  unsigned __int64 v33; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v34[7]; // [rsp+48h] [rbp-38h] BYREF

  v3 = a3;
  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = sub_939680(*(_QWORD **)(a2 + 32), *(_QWORD *)(a2 + 32) + 4LL * *(_QWORD *)(a2 + 40));
    v8 = *(_QWORD **)(a2 + 16);
    v34[0] = v7;
    v33 = sub_BCC330(v8, (__int64)&v8[*(_QWORD *)(a2 + 24)]);
    v9 = (v4 - 1) & sub_BCC270((__int64 *)a2, &v33, v34);
    v10 = (_QWORD *)(v6 + 8LL * v9);
    v11 = *v10;
    if ( *v10 != -4096 )
    {
      v30 = v6;
      v12 = 0;
      v13 = v4 - 1;
      v32 = v10;
      v14 = 1;
      v31 = v3;
      v15 = v9;
      while ( 1 )
      {
        if ( v11 == -8192 )
        {
          if ( !v12 )
            v12 = v32;
        }
        else
        {
          v16 = *(_QWORD *)(a2 + 8);
          if ( v16 == *(_QWORD *)(v11 + 32) )
          {
            v17 = *(const void **)(v11 + 16);
            v18 = *(unsigned int *)(v11 + 12);
            v28 = *(_DWORD *)(v11 + 8) >> 8;
            s2 = *(void **)(v11 + 40);
            if ( !v16
              || (v25 = v12,
                  v26 = *(const void **)(v11 + 16),
                  v19 = memcmp(*(const void **)a2, *(const void **)(v11 + 24), v16),
                  v17 = v26,
                  v12 = v25,
                  !v19) )
            {
              if ( v18 == *(_QWORD *)(a2 + 24) )
              {
                v20 = 8 * v18;
                if ( !v20 || (v27 = v12, v21 = memcmp(*(const void **)(a2 + 16), v17, v20), v12 = v27, !v21) )
                {
                  if ( v28 == *(_QWORD *)(a2 + 40) )
                  {
                    v22 = 4LL * v28;
                    v29 = v12;
                    if ( !v22 || (v23 = memcmp(*(const void **)(a2 + 32), s2, v22), v12 = v29, !v23) )
                    {
                      *v31 = v32;
                      return 1;
                    }
                  }
                }
              }
            }
          }
        }
        v15 = v13 & (v14 + v15);
        v32 = (_QWORD *)(v30 + 8LL * v15);
        v11 = *v32;
        if ( *v32 == -4096 )
          break;
        ++v14;
      }
      v10 = (_QWORD *)(v30 + 8LL * v15);
      v3 = v31;
      if ( v12 )
        v10 = v12;
    }
    *v3 = v10;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
