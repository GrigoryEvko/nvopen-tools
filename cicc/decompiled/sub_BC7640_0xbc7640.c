// Function: sub_BC7640
// Address: 0xbc7640
//
unsigned __int64 __fastcall sub_BC7640(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned __int64 v5; // rbx
  _DWORD *v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rbx
  const void *v17; // rsi
  size_t v18; // rbx
  __int64 v20; // rax
  _QWORD *v21; // rsi
  _QWORD *v22; // rcx
  _BYTE *v23; // rdi
  __int64 v24; // rdx
  size_t v25; // rdx
  size_t v26; // rdx
  _BYTE *v27; // rdi
  _QWORD *v28; // [rsp+8h] [rbp-198h]
  __int64 v29; // [rsp+10h] [rbp-190h]
  unsigned int v33; // [rsp+38h] [rbp-168h]
  unsigned int v34; // [rsp+38h] [rbp-168h]
  _QWORD *v35; // [rsp+40h] [rbp-160h] BYREF
  size_t n; // [rsp+48h] [rbp-158h]
  _QWORD src[2]; // [rsp+50h] [rbp-150h] BYREF
  _QWORD v38[4]; // [rsp+60h] [rbp-140h] BYREF
  __int16 v39; // [rsp+80h] [rbp-120h]
  __int64 *v40; // [rsp+90h] [rbp-110h] BYREF
  __int64 v41; // [rsp+98h] [rbp-108h]
  __int64 v42; // [rsp+A0h] [rbp-100h]
  __int64 v43; // [rsp+A8h] [rbp-F8h] BYREF
  void *dest; // [rsp+B0h] [rbp-F0h]

  sub_2241E40();
  if ( !*(_DWORD *)(a1 + 8) )
  {
LABEL_12:
    v11 = 0;
    return v11 | v29 & 0xFFFFFFFF00000000LL;
  }
  v4 = 0;
  v5 = 0;
  while ( 1 )
  {
    v7 = (_DWORD *)(4 * v5 + *(_QWORD *)a1);
    if ( *v7 == -1 )
      break;
LABEL_5:
    if ( v5 >= a3 )
    {
      v5 = v4 + 1;
      v4 = v5;
      if ( *(_DWORD *)(a1 + 8) <= (unsigned int)v5 )
        goto LABEL_12;
    }
    else
    {
      v8 = 4 * v5 + *(_QWORD *)a1;
      v9 = *(_QWORD *)a4 + 32 * v5;
      LOWORD(dest) = 260;
      v40 = (__int64 *)v9;
      v10 = sub_C83360(&v40, v8, 0, 2, 0, 438);
      v11 = v10;
      if ( v10 )
        goto LABEL_31;
      v12 = *(unsigned int *)(*(_QWORD *)a1 + 4 * v5);
      sub_CB6EE0(&v40, v12, 1, 0, 0);
      if ( *(_DWORD *)(*(_QWORD *)a1 + 4 * v5) == -1 )
      {
        sub_2241E50(&v40, v12, v13, v14, v15);
        sub_CB5B00(&v40);
        v11 = 5;
        goto LABEL_31;
      }
      v16 = a2 + 16 * v5;
      v17 = *(const void **)v16;
      v18 = *(_QWORD *)(v16 + 8);
      if ( v18 > v43 - (__int64)dest )
      {
        sub_CB6200(&v40, v17, v18);
      }
      else if ( v18 )
      {
        memcpy(dest, v17, v18);
        dest = (char *)dest + v18;
      }
      sub_CB5B00(&v40);
      v5 = ++v4;
      if ( *(_DWORD *)(a1 + 8) <= v4 )
        goto LABEL_12;
    }
  }
  v41 = 0;
  v42 = 200;
  v40 = &v43;
  v38[0] = "tmpfile";
  v39 = 259;
  v20 = sub_C85AA0(v38, "txt", 3, v7, &v40, 0);
  v29 = v20;
  v11 = (unsigned int)v20;
  if ( !(_DWORD)v20 )
  {
    v39 = 261;
    v38[0] = v40;
    v38[1] = v41;
    sub_CA0F50(&v35, v38);
    v21 = src;
    v22 = (_QWORD *)(*(_QWORD *)a4 + 32 * v5);
    v23 = (_BYTE *)*v22;
    if ( v35 != src )
    {
      if ( v23 == (_BYTE *)(v22 + 2) )
      {
        *v22 = v35;
        v22[1] = n;
        v22[2] = src[0];
      }
      else
      {
        *v22 = v35;
        v24 = v22[2];
        v22[1] = n;
        v22[2] = src[0];
        if ( v23 )
        {
          v35 = v23;
          src[0] = v24;
          goto LABEL_19;
        }
      }
      v35 = src;
      v21 = src;
      v23 = src;
      goto LABEL_19;
    }
    v25 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *v23 = src[0];
        v26 = n;
        v27 = (_BYTE *)*v22;
        v22[1] = n;
        v27[v26] = 0;
        v23 = v35;
        goto LABEL_19;
      }
      v28 = (_QWORD *)(*(_QWORD *)a4 + 32 * v5);
      memcpy(v23, src, n);
      v22 = v28;
      v25 = n;
      v21 = src;
      v23 = (_BYTE *)*v28;
    }
    v22[1] = v25;
    v23[v25] = 0;
    v23 = v35;
LABEL_19:
    n = 0;
    *v23 = 0;
    if ( v35 != src )
    {
      v21 = (_QWORD *)(src[0] + 1LL);
      j_j___libc_free_0(v35, src[0] + 1LL);
    }
    if ( v40 != &v43 )
      _libc_free(v40, v21);
    goto LABEL_5;
  }
  if ( v40 != &v43 )
  {
    v34 = v20;
    _libc_free(v40, "txt");
    v11 = v34;
  }
LABEL_31:
  if ( v4 )
  {
    v33 = v11;
    sub_BC5E00(*(_QWORD *)a4, *(unsigned int *)(a4 + 8), v4);
    v11 = v33;
  }
  return v11 | v29 & 0xFFFFFFFF00000000LL;
}
