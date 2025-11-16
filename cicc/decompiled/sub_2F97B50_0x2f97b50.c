// Function: sub_2F97B50
// Address: 0x2f97b50
//
void __fastcall sub_2F97B50(__int64 a1, __int64 a2, size_t a3, int a4, __int64 a5, unsigned __int64 a6)
{
  char *v6; // r8
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rbx
  char *v10; // rax
  _QWORD *v11; // rcx
  _QWORD *v12; // rbx
  _QWORD *i; // r14
  _QWORD *j; // r13
  __int64 v15; // rcx
  _QWORD *v16; // rbx
  _QWORD *k; // r14
  _QWORD *m; // r13
  unsigned int *v19; // r15
  __int64 v20; // rbx
  unsigned __int64 v21; // rax
  unsigned int *v22; // rbx
  unsigned int *v23; // rax
  unsigned int *v24; // rsi
  unsigned int *v25; // rsi
  _DWORD *v26; // r15
  char *v27; // r8
  unsigned __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rcx
  unsigned __int64 v31; // r8
  unsigned __int64 v32; // r9
  __int64 v33; // rdi
  int v34; // eax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdi
  int v38; // eax
  unsigned int *v39; // [rsp+0h] [rbp-80h]
  char *v40; // [rsp+0h] [rbp-80h]
  __int64 v42; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+20h] [rbp-60h] BYREF
  __int64 v45; // [rsp+28h] [rbp-58h]
  void *src; // [rsp+30h] [rbp-50h] BYREF
  char *v47; // [rsp+38h] [rbp-48h]
  char *v48; // [rsp+40h] [rbp-40h]

  v6 = 0;
  v8 = (unsigned int)(*(_DWORD *)(a2 + 224) + *(_DWORD *)(a3 + 224));
  v42 = a3;
  src = 0;
  v47 = 0;
  v48 = 0;
  if ( v8 )
  {
    v9 = 4 * v8;
    v10 = (char *)sub_22077B0(v9);
    v6 = v10;
    a3 = v47 - (_BYTE *)src;
    if ( v47 - (_BYTE *)src > 0 )
    {
      v40 = (char *)memmove(v10, src, a3);
      j_j___libc_free_0((unsigned __int64)src);
      v6 = v40;
    }
    src = v6;
    v47 = v6;
    v48 = &v6[v9];
  }
  v11 = *(_QWORD **)(a2 + 80);
  v12 = &v11[4 * *(unsigned int *)(a2 + 88)];
  if ( v11 != v12 )
  {
    for ( i = v11 + 1; ; i += 4 )
    {
      for ( j = (_QWORD *)*i; j != i; j = (_QWORD *)*j )
      {
        while ( 1 )
        {
          a3 = j[2];
          if ( v48 != v6 )
            break;
          sub_B8BBF0((__int64)&src, v6, (_DWORD *)(a3 + 200));
          j = (_QWORD *)*j;
          v6 = v47;
          if ( j == i )
            goto LABEL_14;
        }
        if ( v6 )
        {
          a3 = *(unsigned int *)(a3 + 200);
          *(_DWORD *)v6 = a3;
          v6 = v47;
        }
        v6 += 4;
        v47 = v6;
      }
LABEL_14:
      if ( v12 == i + 3 )
        break;
    }
  }
  v15 = *(_QWORD *)(v42 + 80);
  v16 = (_QWORD *)(v15 + 32LL * *(unsigned int *)(v42 + 88));
  if ( v16 != (_QWORD *)v15 )
  {
    for ( k = (_QWORD *)(v15 + 8); ; k += 4 )
    {
      for ( m = (_QWORD *)*k; m != k; m = (_QWORD *)*m )
      {
        while ( 1 )
        {
          a3 = m[2];
          if ( v48 != v6 )
            break;
          sub_B8BBF0((__int64)&src, v6, (_DWORD *)(a3 + 200));
          m = (_QWORD *)*m;
          v6 = v47;
          if ( m == k )
            goto LABEL_25;
        }
        if ( v6 )
        {
          a3 = *(unsigned int *)(a3 + 200);
          *(_DWORD *)v6 = a3;
          v6 = v47;
        }
        v6 += 4;
        v47 = v6;
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
    v39 = (unsigned int *)v6;
    _BitScanReverse64(&v21, (v6 - (_BYTE *)src) >> 2);
    sub_2F910F0((char *)src, v6, 2LL * (int)(63 - (v21 ^ 0x3F)));
    if ( v20 <= 64 )
    {
      sub_2F90F70(v19, v39);
    }
    else
    {
      v22 = v19 + 16;
      sub_2F90F70(v19, v19 + 16);
      if ( v19 + 16 != v39 )
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
          if ( v22 == v39 )
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
            if ( v22 == v39 )
              goto LABEL_37;
          }
        }
      }
    }
LABEL_37:
    v6 = v47;
  }
  v26 = *(_DWORD **)(a1 + 2904);
  v27 = &v6[-4 * a4];
  v28 = *(_QWORD *)(a1 + 48) + ((unsigned __int64)*(unsigned int *)v27 << 8);
  if ( v26 )
  {
    if ( *(_DWORD *)(v28 + 200) < v26[50] )
    {
      v45 = 0;
      v44 = v28 | 6;
      v33 = *(_QWORD *)v28;
      if ( (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v28 + 68LL) - 1 <= 1
        && (*(_BYTE *)(*(_QWORD *)(v33 + 32) + 64LL) & 0x10) != 0
        || ((v34 = *(_DWORD *)(v33 + 44), (v34 & 4) == 0) && (v34 & 8) != 0
          ? (LOBYTE(v35) = sub_2E88A90(v33, 0x100000, 1))
          : (v35 = (*(_QWORD *)(*(_QWORD *)(v33 + 16) + 24LL) >> 20) & 1LL),
            (_BYTE)v35) )
      {
        v37 = *(_QWORD *)v26;
        if ( (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v26 + 68LL) - 1 > 1
          || (LODWORD(v36) = 1, (*(_BYTE *)(*(_QWORD *)(v37 + 32) + 64LL) & 8) == 0) )
        {
          v38 = *(_DWORD *)(v37 + 44);
          if ( (v38 & 4) != 0 || (v38 & 8) == 0 )
            v36 = (*(_QWORD *)(*(_QWORD *)(v37 + 16) + 24LL) >> 19) & 1LL;
          else
            LOBYTE(v36) = sub_2E88A90(v37, 0x80000, 1);
          LODWORD(v36) = (unsigned __int8)v36;
        }
      }
      else
      {
        LODWORD(v36) = 0;
      }
      HIDWORD(v45) = v36;
      sub_2F8F1B0((__int64)v26, (__int64)&v44, 1u, v15, (unsigned __int64)v27, a6);
      *(_QWORD *)(a1 + 2904) = v28;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 2904) = v28;
  }
  sub_2F97470(a1, a2, a3, v15, (unsigned __int64)v27, a6);
  sub_2F97470(a1, v42, v29, v30, v31, v32);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
}
