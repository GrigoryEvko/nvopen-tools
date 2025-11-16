// Function: sub_B3F3B0
// Address: 0xb3f3b0
//
int __fastcall sub_B3F3B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  bool v6; // zf
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // r14
  _QWORD *v10; // rax
  char *v11; // r14
  char **v12; // r9
  char *v13; // rax
  char *v14; // rdi
  char *v15; // rdx
  char *v16; // r10
  __int16 v17; // r9
  size_t v18; // rdx
  size_t v19; // rdx
  __int16 v21; // [rsp+4h] [rbp-BCh]
  char *v22; // [rsp+8h] [rbp-B8h]
  char *v23; // [rsp+8h] [rbp-B8h]
  char **v24; // [rsp+8h] [rbp-B8h]
  __int64 v25; // [rsp+28h] [rbp-98h] BYREF
  void *v26; // [rsp+30h] [rbp-90h]
  size_t v27; // [rsp+38h] [rbp-88h]
  _QWORD v28[2]; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int16 v29; // [rsp+50h] [rbp-70h]
  void *s1; // [rsp+60h] [rbp-60h]
  size_t n; // [rsp+68h] [rbp-58h]
  _QWORD v32[2]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v33; // [rsp+80h] [rbp-40h]

  v3 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  s1 = v32;
  n = 0;
  LOBYTE(v32[0]) = 0;
  v33 = 0;
  if ( v6 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = 48LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v7 = a1 + 16;
    v8 = 1536;
  }
  v9 = v7 + v8;
  if ( v9 != v7 )
  {
    do
    {
      if ( v7 )
      {
        *(_QWORD *)v7 = v7 + 16;
        sub_B3AE60((__int64 *)v7, s1, (__int64)s1 + n);
        *(_WORD *)(v7 + 32) = v33;
      }
      v7 += 48;
    }
    while ( v9 != v7 );
    if ( s1 != v32 )
      j_j___libc_free_0(s1, v32[0] + 1LL);
  }
  LOBYTE(v28[0]) = 0;
  v26 = v28;
  v29 = 0;
  v10 = v32;
  v27 = 0;
  s1 = v32;
  n = 0;
  LOBYTE(v32[0]) = 0;
  v33 = 257;
  if ( a2 != a3 )
  {
    v11 = (char *)(a2 + 16);
    for ( LODWORD(v10) = 0; ; LODWORD(v10) = v29 )
    {
      v17 = *(_WORD *)(v3 + 32);
      if ( v17 == (_WORD)v10 )
      {
        v16 = *(char **)v3;
        if ( !*(_BYTE *)(v3 + 32) || *(_BYTE *)(v3 + 33) )
          goto LABEL_17;
        if ( v27 == *(_QWORD *)(v3 + 8) )
        {
          v21 = *(_WORD *)(v3 + 32);
          if ( !v27 )
            goto LABEL_17;
          v23 = *(char **)v3;
          LODWORD(v10) = memcmp(v26, v16, v27);
          v16 = v23;
          if ( !(_DWORD)v10 )
            goto LABEL_17;
          v17 = v21;
        }
        if ( v17 != v33 || (v18 = n, n != *(_QWORD *)(v3 + 8)) )
        {
LABEL_12:
          sub_B3C4F0(a1, v3, &v25);
          v12 = (char **)v25;
          v13 = *(char **)v3;
          v14 = *(char **)v25;
          if ( v11 == *(char **)v3 )
          {
            v19 = *(_QWORD *)(v3 + 8);
            if ( v19 )
            {
              if ( v19 == 1 )
              {
                *v14 = *(_BYTE *)(v3 + 16);
                v19 = *(_QWORD *)(v3 + 8);
                v14 = *v12;
              }
              else
              {
                v24 = (char **)v25;
                memcpy(v14, v11, v19);
                v12 = v24;
                v19 = *(_QWORD *)(v3 + 8);
                v14 = *v24;
              }
            }
            v12[1] = (char *)v19;
            v14[v19] = 0;
            v14 = *(char **)v3;
            goto LABEL_16;
          }
          if ( v14 == (char *)(v25 + 16) )
          {
            *(_QWORD *)v25 = v13;
            v12[1] = *(char **)(v3 + 8);
            v12[2] = *(char **)(v3 + 16);
          }
          else
          {
            *(_QWORD *)v25 = v13;
            v15 = v12[2];
            v12[1] = *(char **)(v3 + 8);
            v12[2] = *(char **)(v3 + 16);
            if ( v14 )
            {
              *(_QWORD *)v3 = v14;
              *(_QWORD *)(v3 + 16) = v15;
LABEL_16:
              *(_QWORD *)(v3 + 8) = 0;
              *v14 = 0;
              *((_BYTE *)v12 + 32) = *(_BYTE *)(v3 + 32);
              *((_BYTE *)v12 + 33) = *(_BYTE *)(v3 + 33);
              *(_DWORD *)(v25 + 40) = *(_DWORD *)(v3 + 40);
              LODWORD(v10) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              *(_DWORD *)(a1 + 8) = (_DWORD)v10;
              v16 = *(char **)v3;
              goto LABEL_17;
            }
          }
          *(_QWORD *)v3 = v11;
          v14 = v11;
          goto LABEL_16;
        }
      }
      else
      {
        if ( v17 != v33 )
          goto LABEL_12;
        v16 = *(char **)v3;
        if ( !*(_BYTE *)(v3 + 32) || *(_BYTE *)(v3 + 33) )
          goto LABEL_17;
        v18 = n;
        if ( n != *(_QWORD *)(v3 + 8) )
          goto LABEL_12;
      }
      if ( v18 )
      {
        v22 = v16;
        LODWORD(v10) = memcmp(s1, v16, v18);
        v16 = v22;
        if ( (_DWORD)v10 )
          goto LABEL_12;
      }
LABEL_17:
      if ( v16 != v11 )
        LODWORD(v10) = j_j___libc_free_0(v16, *(_QWORD *)(v3 + 16) + 1LL);
      v3 += 48;
      v11 += 48;
      if ( a3 == v3 )
      {
        if ( s1 != v32 )
          LODWORD(v10) = j_j___libc_free_0(s1, v32[0] + 1LL);
        if ( v26 != v28 )
          LODWORD(v10) = j_j___libc_free_0(v26, v28[0] + 1LL);
        return (int)v10;
      }
    }
  }
  return (int)v10;
}
