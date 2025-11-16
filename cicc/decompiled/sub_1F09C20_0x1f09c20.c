// Function: sub_1F09C20
// Address: 0x1f09c20
//
__int64 __fastcall sub_1F09C20(__int64 a1, __int64 a2, size_t a3, int a4, __int64 a5, unsigned __int64 a6)
{
  char *v6; // r8
  __int64 v8; // rbx
  __int64 v9; // rbx
  char *v10; // rax
  _QWORD *v11; // rcx
  _QWORD *v12; // r15
  _QWORD *i; // r14
  _QWORD *j; // r13
  __int64 v15; // rcx
  _QWORD *v16; // r15
  _QWORD *k; // r14
  _QWORD *m; // r13
  unsigned int *v19; // r15
  __int64 v20; // rbx
  unsigned __int64 v21; // rax
  unsigned int *v22; // rbx
  unsigned int *v23; // rax
  unsigned int *v24; // rsi
  unsigned int *v25; // rsi
  __int64 v26; // r15
  char *v27; // r8
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rcx
  int v31; // r8d
  unsigned __int64 v32; // r9
  __int64 result; // rax
  __int64 v34; // rdi
  __int64 v35; // rax
  __int16 v36; // dx
  bool v37; // al
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rdx
  __int16 v41; // ax
  unsigned int *v42; // [rsp+0h] [rbp-80h]
  char *v43; // [rsp+0h] [rbp-80h]
  __int64 v45; // [rsp+10h] [rbp-70h]
  __int64 v47; // [rsp+20h] [rbp-60h] BYREF
  __int64 v48; // [rsp+28h] [rbp-58h]
  void *src; // [rsp+30h] [rbp-50h] BYREF
  char *v50; // [rsp+38h] [rbp-48h]
  char *v51; // [rsp+40h] [rbp-40h]

  v6 = 0;
  v8 = (unsigned int)(*(_DWORD *)(a2 + 56) + *(_DWORD *)(a3 + 56));
  v45 = a3;
  src = 0;
  v50 = 0;
  v51 = 0;
  if ( v8 )
  {
    v9 = 4 * v8;
    v10 = (char *)sub_22077B0(v9);
    v6 = v10;
    a3 = v50 - (_BYTE *)src;
    if ( v50 - (_BYTE *)src > 0 )
    {
      v43 = (char *)memmove(v10, src, a3);
      j_j___libc_free_0(src, v51 - (_BYTE *)src);
      v6 = v43;
    }
    src = v6;
    v50 = v6;
    v51 = &v6[v9];
  }
  v11 = *(_QWORD **)(a2 + 32);
  v12 = *(_QWORD **)(a2 + 40);
  if ( v11 != v12 )
  {
    for ( i = v11 + 1; ; i += 4 )
    {
      for ( j = (_QWORD *)*i; j != i; j = (_QWORD *)*j )
      {
        while ( 1 )
        {
          a3 = j[2];
          if ( v6 != v51 )
            break;
          sub_B8BBF0((__int64)&src, v6, (_DWORD *)(a3 + 192));
          j = (_QWORD *)*j;
          v6 = v50;
          if ( j == i )
            goto LABEL_14;
        }
        if ( v6 )
        {
          a3 = *(unsigned int *)(a3 + 192);
          *(_DWORD *)v6 = a3;
          v6 = v50;
        }
        v6 += 4;
        v50 = v6;
      }
LABEL_14:
      if ( v12 == i + 3 )
        break;
    }
  }
  v15 = *(_QWORD *)(v45 + 32);
  v16 = *(_QWORD **)(v45 + 40);
  if ( v16 != (_QWORD *)v15 )
  {
    for ( k = (_QWORD *)(v15 + 8); ; k += 4 )
    {
      for ( m = (_QWORD *)*k; m != k; m = (_QWORD *)*m )
      {
        while ( 1 )
        {
          a3 = m[2];
          if ( v51 != v6 )
            break;
          sub_B8BBF0((__int64)&src, v6, (_DWORD *)(a3 + 192));
          m = (_QWORD *)*m;
          v6 = v50;
          if ( m == k )
            goto LABEL_25;
        }
        if ( v6 )
        {
          a3 = *(unsigned int *)(a3 + 192);
          *(_DWORD *)v6 = a3;
          v6 = v50;
        }
        v6 += 4;
        v50 = v6;
      }
LABEL_25:
      v15 = (__int64)(k + 3);
      if ( v16 == k + 3 )
        break;
    }
  }
  v19 = (unsigned int *)src;
  if ( src != v6 )
  {
    v20 = v6 - (_BYTE *)src;
    v42 = (unsigned int *)v6;
    _BitScanReverse64(&v21, (v6 - (_BYTE *)src) >> 2);
    sub_1F03720((char *)src, v6, 2LL * (int)(63 - (v21 ^ 0x3F)));
    if ( v20 <= 64 )
    {
      sub_1F03670(v19, v42);
    }
    else
    {
      v22 = v19 + 16;
      sub_1F03670(v19, v19 + 16);
      if ( v19 + 16 != v42 )
      {
        v15 = *v22;
        a3 = v19[15];
        v23 = v19 + 15;
        if ( (unsigned int)v15 >= (unsigned int)a3 )
          goto LABEL_34;
        while ( 1 )
        {
          do
          {
            v23[1] = a3;
            v24 = v23;
            a3 = *--v23;
          }
          while ( (unsigned int)v15 < (unsigned int)a3 );
          ++v22;
          *v24 = v15;
          if ( v22 == v42 )
            break;
          while ( 1 )
          {
            v15 = *v22;
            a3 = *(v22 - 1);
            v23 = v22 - 1;
            if ( (unsigned int)v15 < (unsigned int)a3 )
              break;
LABEL_34:
            v25 = v22++;
            *v25 = v15;
            if ( v22 == v42 )
              goto LABEL_37;
          }
        }
      }
    }
LABEL_37:
    v6 = v50;
  }
  v26 = *(_QWORD *)(a1 + 1984);
  v27 = &v6[-4 * a4];
  v28 = *(_QWORD *)(a1 + 48) + 272LL * *(unsigned int *)v27;
  if ( v26 )
  {
    if ( *(_DWORD *)(v28 + 192) < *(_DWORD *)(v26 + 192) )
    {
      v48 = 0;
      v34 = *(_QWORD *)(v28 + 8);
      v47 = v28 | 6;
      v35 = *(_QWORD *)(v34 + 16);
      if ( *(_WORD *)v35 == 1 && (*(_BYTE *)(*(_QWORD *)(v34 + 32) + 64LL) & 0x10) != 0
        || ((v36 = *(_WORD *)(v34 + 46), (v36 & 4) == 0) && (v36 & 8) != 0
          ? (v37 = sub_1E15D00(v34, 0x20000u, 1))
          : (v37 = (*(_QWORD *)(v35 + 8) & 0x20000LL) != 0),
            v37) )
      {
        v39 = *(_QWORD *)(v26 + 8);
        v40 = *(_QWORD *)(v39 + 16);
        if ( *(_WORD *)v40 != 1 || (v15 = *(_QWORD *)(v39 + 32), LODWORD(v38) = 1, (*(_BYTE *)(v15 + 64) & 8) == 0) )
        {
          v41 = *(_WORD *)(v39 + 46);
          if ( (v41 & 4) != 0 || (v41 & 8) == 0 )
            v38 = (*(_QWORD *)(v40 + 8) >> 16) & 1LL;
          else
            LOBYTE(v38) = sub_1E15D00(v39, 0x10000u, 1);
          LODWORD(v38) = (unsigned __int8)v38;
        }
      }
      else
      {
        LODWORD(v38) = 0;
      }
      HIDWORD(v48) = v38;
      sub_1F01A00(v26, (__int64)&v47, 1, v15, (int)v27, a6);
      *(_QWORD *)(a1 + 1984) = v28;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 1984) = v28;
  }
  sub_1F09540(a1, a2, a3, v15, (int)v27, a6);
  result = sub_1F09540(a1, v45, v29, v30, v31, v32);
  if ( src )
    return j_j___libc_free_0(src, v51 - (_BYTE *)src);
  return result;
}
