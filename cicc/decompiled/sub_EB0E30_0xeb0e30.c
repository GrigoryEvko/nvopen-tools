// Function: sub_EB0E30
// Address: 0xeb0e30
//
__int64 __fastcall sub_EB0E30(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // rax
  bool v6; // cc
  _QWORD *v7; // rax
  __int64 v8; // rax
  _BYTE *v9; // r15
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rax
  bool v12; // zf
  int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rcx
  _BYTE *v17; // r9
  __int64 v18; // r13
  __int64 v19; // rax
  unsigned int *v20; // rdi
  __int64 v21; // rcx
  size_t v22; // rsi
  __int64 v23; // rdx
  unsigned int *v24; // rdi
  size_t v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdx
  size_t v29; // rdx
  size_t v30; // rdx
  _BYTE *v31; // [rsp+0h] [rbp-70h]
  _QWORD *v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  int v34; // [rsp+1Ch] [rbp-54h] BYREF
  unsigned int *v35; // [rsp+20h] [rbp-50h] BYREF
  size_t n; // [rsp+28h] [rbp-48h]
  _QWORD src[8]; // [rsp+30h] [rbp-40h] BYREF

  sub_EABFE0(a1);
  v5 = sub_ECD7B0(a1);
  v6 = *(_DWORD *)(v5 + 32) <= 0x40u;
  v7 = *(_QWORD **)(v5 + 24);
  if ( !v6 )
    v7 = (_QWORD *)*v7;
  v32 = v7;
  sub_EABFE0(a1);
  v8 = sub_ECD7B0(a1);
  v9 = *(_BYTE **)(v8 + 8);
  v10 = *(_QWORD *)(v8 + 16);
  sub_EABFE0(a1);
  if ( a3 )
  {
    if ( v10 )
    {
      v11 = v10 - 2;
      if ( v11 <= --v10 )
        v10 = v11;
      ++v9;
    }
    v12 = *(_BYTE *)(a1 + 520) == 0;
    *(_QWORD *)(a1 + 504) = a2;
    *(_QWORD *)(a1 + 480) = v9;
    *(_QWORD *)(a1 + 496) = v32;
    v13 = *(_DWORD *)(a1 + 304);
    *(_QWORD *)(a1 + 488) = v10;
    *(_DWORD *)(a1 + 512) = v13;
    if ( v12 )
    {
      v14 = *(_QWORD *)(a1 + 224);
      *(_BYTE *)(a1 + 520) = 1;
      if ( *(_BYTE *)(v14 + 1793) )
      {
        if ( !*(_DWORD *)(v14 + 1796) )
        {
          v15 = *(_QWORD *)(v14 + 1744);
          v34 = 0;
          v16 = *(_QWORD *)(v14 + 1536);
          v17 = *(_BYTE **)(v14 + 1528);
          if ( v15 )
          {
            do
            {
              v18 = v15;
              v15 = *(_QWORD *)(v15 + 16);
            }
            while ( v15 );
            if ( v18 != v14 + 1736 && !*(_DWORD *)(v18 + 32) )
            {
LABEL_16:
              v35 = (unsigned int *)src;
              sub_EA2A30((__int64 *)&v35, v17, (__int64)&v17[v16]);
              v20 = *(unsigned int **)(v18 + 440);
              if ( v35 == (unsigned int *)src )
              {
                v30 = n;
                if ( n )
                {
                  if ( n == 1 )
                    *(_BYTE *)v20 = src[0];
                  else
                    memcpy(v20, src, n);
                  v30 = n;
                  v20 = *(unsigned int **)(v18 + 440);
                }
                *(_QWORD *)(v18 + 448) = v30;
                *((_BYTE *)v20 + v30) = 0;
                v20 = v35;
              }
              else
              {
                v21 = src[0];
                v22 = n;
                if ( v20 == (unsigned int *)(v18 + 456) )
                {
                  *(_QWORD *)(v18 + 440) = v35;
                  *(_QWORD *)(v18 + 448) = v22;
                  *(_QWORD *)(v18 + 456) = v21;
                }
                else
                {
                  v23 = *(_QWORD *)(v18 + 456);
                  *(_QWORD *)(v18 + 440) = v35;
                  *(_QWORD *)(v18 + 448) = v22;
                  *(_QWORD *)(v18 + 456) = v21;
                  if ( v20 )
                  {
                    v35 = v20;
                    src[0] = v23;
                    goto LABEL_20;
                  }
                }
                v35 = (unsigned int *)src;
                v20 = (unsigned int *)src;
              }
LABEL_20:
              n = 0;
              *(_BYTE *)v20 = 0;
              if ( v35 != (unsigned int *)src )
                j_j___libc_free_0(v35, src[0] + 1LL);
              v35 = (unsigned int *)src;
              sub_EA2A30((__int64 *)&v35, v9, (__int64)&v9[v10]);
              v24 = *(unsigned int **)(v18 + 472);
              if ( v35 == (unsigned int *)src )
              {
                v29 = n;
                if ( n )
                {
                  if ( n == 1 )
                    *(_BYTE *)v24 = src[0];
                  else
                    memcpy(v24, src, n);
                  v29 = n;
                  v24 = *(unsigned int **)(v18 + 472);
                }
                *(_QWORD *)(v18 + 480) = v29;
                *((_BYTE *)v24 + v29) = 0;
                v24 = v35;
                goto LABEL_26;
              }
              v25 = n;
              v26 = src[0];
              if ( v24 == (unsigned int *)(v18 + 488) )
              {
                *(_QWORD *)(v18 + 472) = v35;
                *(_QWORD *)(v18 + 480) = v25;
                *(_QWORD *)(v18 + 488) = v26;
              }
              else
              {
                v27 = *(_QWORD *)(v18 + 488);
                *(_QWORD *)(v18 + 472) = v35;
                *(_QWORD *)(v18 + 480) = v25;
                *(_QWORD *)(v18 + 488) = v26;
                if ( v24 )
                {
                  v35 = v24;
                  src[0] = v27;
LABEL_26:
                  n = 0;
                  *(_BYTE *)v24 = 0;
                  if ( v35 != (unsigned int *)src )
                    j_j___libc_free_0(v35, src[0] + 1LL);
                  *(_DWORD *)(v18 + 504) = 0;
                  *(_BYTE *)(v18 + 524) = 0;
                  *(_BYTE *)(v18 + 544) = 0;
                  *(_BYTE *)(v18 + 553) = 0;
                  return 0;
                }
              }
              v35 = (unsigned int *)src;
              v24 = (unsigned int *)src;
              goto LABEL_26;
            }
          }
          else
          {
            v18 = v14 + 1736;
          }
          v31 = *(_BYTE **)(v14 + 1528);
          v33 = *(_QWORD *)(v14 + 1536);
          v35 = (unsigned int *)&v34;
          v19 = sub_EAA600((_QWORD *)(v14 + 1728), v18, &v35);
          v17 = v31;
          v16 = v33;
          v18 = v19;
          goto LABEL_16;
        }
      }
    }
  }
  return 0;
}
