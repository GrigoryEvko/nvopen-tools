// Function: sub_26CB210
// Address: 0x26cb210
//
__int64 __fastcall sub_26CB210(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  unsigned __int64 v10; // r13
  _QWORD *v11; // r15
  _BYTE *v12; // rsi
  _QWORD *v13; // r14
  char *v14; // r13
  char *v15; // r14
  unsigned __int64 v16; // rax
  char *v17; // rbx
  char *v18; // rdi
  __int64 v20; // r14
  __int64 v21; // rdi
  unsigned int *v22; // r13
  __int64 v23; // rax
  _QWORD *v24; // rdi
  __int64 v25; // rax
  unsigned int *v26; // rsi
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rdx
  _QWORD *v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned int v33; // ecx
  __int64 v34; // rdx
  __int64 v35; // r15
  __int64 i; // r13
  _BYTE *v37; // rsi
  __int64 v38; // rax
  char *v39; // r13
  char *v40; // r14
  unsigned __int64 v41; // rax
  char *v42; // rbx
  char *v43; // rdi
  _QWORD *v44; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v45; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v46; // [rsp+28h] [rbp-48h]

  v7 = sub_B10CD0(a3 + 48);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  if ( !v7 )
    return a1;
  v9 = v7;
  if ( !unk_4F838D3 )
  {
    v20 = sub_26CAC90(a2, a3, v8);
    if ( !v20 )
      return a1;
    v21 = v9;
    v22 = (unsigned int *)&v45;
    v23 = sub_C1B090(v21, 0);
    *a4 = 0;
    v45 = (_QWORD *)v23;
    v24 = *(_QWORD **)(v20 + 168);
    if ( v24 && (v25 = sub_C1BA30(v24, (__int64)&v45)) != 0 )
      v26 = (unsigned int *)(v25 + 16);
    else
      v26 = (unsigned int *)&v45;
    v27 = sub_26C2A80(v20 + 72, v26);
    if ( v27 != v20 + 80 )
    {
      v28 = *(_QWORD **)(v27 + 64);
      if ( v28 )
      {
        v29 = *a4;
        do
        {
          v29 += v28[3];
          *a4 = v29;
          v28 = (_QWORD *)*v28;
        }
        while ( v28 );
      }
    }
    v30 = *(_QWORD **)(v20 + 168);
    if ( v30 )
    {
      v31 = sub_C1BA30(v30, (__int64)&v45);
      if ( v31 )
        v22 = (unsigned int *)(v31 + 16);
    }
    v32 = *(_QWORD *)(v20 + 136);
    if ( !v32 )
      return a1;
    v33 = *v22;
    v34 = v20 + 128;
    while ( 1 )
    {
      while ( *(_DWORD *)(v32 + 32) < v33 )
      {
        v32 = *(_QWORD *)(v32 + 24);
LABEL_35:
        if ( !v32 )
        {
LABEL_36:
          if ( v20 + 128 != v34
            && *(_DWORD *)(v34 + 32) <= v33
            && (*(_DWORD *)(v34 + 32) != v33 || v22[1] >= *(_DWORD *)(v34 + 36))
            && *(_QWORD *)(v34 + 80) )
          {
            v35 = *(_QWORD *)(v34 + 64);
            for ( i = v34 + 48; i != v35; v35 = sub_220EF30(v35) )
            {
              v38 = sub_EF9210((_QWORD *)(v35 + 48));
              v44 = (_QWORD *)(v35 + 48);
              *a4 += v38;
              v37 = *(_BYTE **)(a1 + 8);
              if ( v37 == *(_BYTE **)(a1 + 16) )
              {
                sub_26C5540(a1, v37, &v44);
              }
              else
              {
                if ( v37 )
                {
                  *(_QWORD *)v37 = v35 + 48;
                  v37 = *(_BYTE **)(a1 + 8);
                }
                *(_QWORD *)(a1 + 8) = v37 + 8;
              }
            }
            v39 = *(char **)(a1 + 8);
            v40 = *(char **)a1;
            if ( v39 != *(char **)a1 )
            {
              _BitScanReverse64(&v41, (v39 - v40) >> 3);
              sub_26BF480(*(char **)a1, *(char **)(a1 + 8), 2LL * (int)(63 - (v41 ^ 0x3F)));
              if ( v39 - v40 <= 128 )
              {
                sub_26BE700(v40, v39);
              }
              else
              {
                v42 = v40 + 128;
                sub_26BE700(v40, v40 + 128);
                if ( v39 != v40 + 128 )
                {
                  do
                  {
                    v43 = v42;
                    v42 += 8;
                    sub_26BE3E0(v43);
                  }
                  while ( v39 != v42 );
                }
              }
            }
          }
          return a1;
        }
      }
      if ( *(_DWORD *)(v32 + 32) == v33 && *(_DWORD *)(v32 + 36) < v22[1] )
      {
        v32 = *(_QWORD *)(v32 + 24);
        goto LABEL_35;
      }
      v34 = v32;
      v32 = *(_QWORD *)(v32 + 16);
      if ( !v32 )
        goto LABEL_36;
    }
  }
  sub_317F220(&v45, *(_QWORD *)(a2 + 1512), v7);
  v10 = v46;
  v11 = v45;
  if ( v45 != (_QWORD *)v46 )
  {
    *a4 = 0;
    do
    {
      while ( 1 )
      {
        v13 = (_QWORD *)*v11;
        v44 = (_QWORD *)*v11;
        *a4 += sub_EF9210(v44);
        v12 = *(_BYTE **)(a1 + 8);
        if ( v12 != *(_BYTE **)(a1 + 16) )
          break;
        ++v11;
        sub_26C32C0(a1, v12, &v44);
        if ( (_QWORD *)v10 == v11 )
          goto LABEL_10;
      }
      if ( v12 )
      {
        *(_QWORD *)v12 = v13;
        v12 = *(_BYTE **)(a1 + 8);
      }
      ++v11;
      *(_QWORD *)(a1 + 8) = v12 + 8;
    }
    while ( (_QWORD *)v10 != v11 );
LABEL_10:
    v14 = *(char **)(a1 + 8);
    v15 = *(char **)a1;
    if ( v14 != *(char **)a1 )
    {
      _BitScanReverse64(&v16, (v14 - v15) >> 3);
      sub_26BF480(*(char **)a1, *(char **)(a1 + 8), 2LL * (int)(63 - (v16 ^ 0x3F)));
      if ( v14 - v15 <= 128 )
      {
        sub_26BE700(v15, v14);
      }
      else
      {
        v17 = v15 + 128;
        sub_26BE700(v15, v15 + 128);
        if ( v14 != v15 + 128 )
        {
          do
          {
            v18 = v17;
            v17 += 8;
            sub_26BE3E0(v18);
          }
          while ( v14 != v17 );
        }
      }
    }
    v10 = (unsigned __int64)v45;
  }
  if ( v10 )
    j_j___libc_free_0(v10);
  return a1;
}
