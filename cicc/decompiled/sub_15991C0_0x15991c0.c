// Function: sub_15991C0
// Address: 0x15991c0
//
__int64 __fastcall sub_15991C0(char *src, size_t n, __int64 **a3)
{
  char *v5; // rax
  __int64 v6; // r15
  unsigned int v7; // r9d
  _QWORD *v8; // r10
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v12; // rax
  unsigned int v13; // r9d
  _QWORD *v14; // r10
  _QWORD *v15; // r8
  _BYTE *v16; // rdi
  __int64 *v17; // rdx
  __int64 *v18; // rdx
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rax
  _BYTE *v22; // rax
  __int64 *v23; // rbx
  _QWORD *v24; // [rsp+0h] [rbp-50h]
  _QWORD *v25; // [rsp+0h] [rbp-50h]
  unsigned int v26; // [rsp+8h] [rbp-48h]
  _QWORD *v27; // [rsp+8h] [rbp-48h]
  _QWORD *v28; // [rsp+8h] [rbp-48h]
  _QWORD *v29; // [rsp+10h] [rbp-40h]
  unsigned int v30; // [rsp+10h] [rbp-40h]
  unsigned int v31; // [rsp+18h] [rbp-38h]

  if ( src != &src[n] )
  {
    v5 = src;
    while ( !*v5 )
    {
      if ( &src[n] == ++v5 )
        return sub_1598F00(a3);
    }
    v6 = **a3;
    v7 = sub_16D19C0(v6 + 1712, src, n);
    v8 = (_QWORD *)(*(_QWORD *)(v6 + 1712) + 8LL * v7);
    v9 = *v8;
    if ( *v8 )
    {
      if ( v9 != -8 )
        goto LABEL_7;
      --*(_DWORD *)(v6 + 1728);
    }
    v24 = v8;
    v26 = v7;
    v12 = malloc(n + 17);
    v13 = v26;
    v14 = v24;
    v15 = (_QWORD *)v12;
    if ( !v12 )
    {
      if ( n == -17 )
      {
        v21 = malloc(1u);
        v13 = v26;
        v14 = v24;
        v15 = 0;
        if ( v21 )
        {
          v16 = (_BYTE *)(v21 + 16);
          v15 = (_QWORD *)v21;
          goto LABEL_31;
        }
      }
      v25 = v15;
      v28 = v14;
      v30 = v13;
      sub_16BD1C0("Allocation failed");
      v13 = v30;
      v14 = v28;
      v15 = v25;
    }
    v16 = v15 + 2;
    if ( n + 1 <= 1 )
    {
LABEL_17:
      v16[n] = 0;
      *v15 = n;
      v15[1] = 0;
      *v14 = v15;
      ++*(_DWORD *)(v6 + 1724);
      v17 = (__int64 *)(*(_QWORD *)(v6 + 1712) + 8LL * (unsigned int)sub_16D1CD0(v6 + 1712, v13));
      v9 = *v17;
      if ( !*v17 || v9 == -8 )
      {
        v18 = v17 + 1;
        do
        {
          do
            v9 = *v18++;
          while ( v9 == -8 );
        }
        while ( !v9 );
      }
LABEL_7:
      v10 = *(_QWORD *)(v9 + 8);
      if ( v10 )
      {
        while ( *(__int64 ***)v10 != a3 )
        {
          if ( !*(_QWORD *)(v10 + 32) )
          {
            v23 = (__int64 *)(v10 + 32);
            goto LABEL_24;
          }
          v10 = *(_QWORD *)(v10 + 32);
        }
        return v10;
      }
      v23 = (__int64 *)(v9 + 8);
LABEL_24:
      v19 = v9 + 16;
      if ( *((_BYTE *)a3 + 8) == 14 )
      {
        v10 = sub_1648A60(40, 0);
        if ( !v10 )
          goto LABEL_28;
        v20 = 11;
      }
      else
      {
        v10 = sub_1648A60(40, 0);
        if ( !v10 )
        {
LABEL_28:
          *v23 = v10;
          return v10;
        }
        v20 = 12;
      }
      sub_1648CB0(v10, a3, v20);
      *(_QWORD *)(v10 + 24) = v19;
      *(_DWORD *)(v10 + 20) &= 0xF0000000;
      *(_QWORD *)(v10 + 32) = 0;
      goto LABEL_28;
    }
LABEL_31:
    v27 = v15;
    v29 = v14;
    v31 = v13;
    v22 = memcpy(v16, src, n);
    v15 = v27;
    v14 = v29;
    v13 = v31;
    v16 = v22;
    goto LABEL_17;
  }
  return sub_1598F00(a3);
}
