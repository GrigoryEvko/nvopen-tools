// Function: sub_183FAF0
// Address: 0x183faf0
//
__int64 __fastcall sub_183FAF0(size_t a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // r14
  __int64 result; // rax
  __int64 v7; // r15
  __int64 v8; // rdx
  unsigned __int64 v9; // r15
  unsigned __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdx
  _BYTE *v14; // rax
  _BYTE *v15; // rsi
  int v16; // r12d
  unsigned __int64 v17; // rbx
  __int64 v18; // rax
  char *v19; // r15
  size_t v20; // r13
  char *v21; // rbx
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rdx
  _BYTE *v25; // rax
  int v26; // ebx
  unsigned __int64 v27; // r12
  __int64 v28; // rax
  char *v29; // r15
  size_t v30; // r13
  char *v31; // r12
  __int64 v32; // rsi
  unsigned __int64 v33; // [rsp+18h] [rbp-98h] BYREF
  char v34[8]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v35; // [rsp+28h] [rbp-88h]
  __int64 v36; // [rsp+38h] [rbp-78h]
  char v37[8]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v38; // [rsp+48h] [rbp-68h]
  __int64 v39; // [rsp+58h] [rbp-58h]
  unsigned __int64 v40; // [rsp+60h] [rbp-50h] BYREF
  __int64 v41; // [rsp+68h] [rbp-48h]
  __int64 v42; // [rsp+70h] [rbp-40h]
  __int64 v43; // [rsp+78h] [rbp-38h]

  v4 = a3;
  result = (unsigned int)*(unsigned __int8 *)(a2 + 16) - 24;
  if ( *(_BYTE *)(a2 + 16) == 55 )
  {
    v24 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v24 + 16) != 3 )
      return result;
    v10 = v24 & 0xFFFFFFFFFFFFFFF9LL | 4;
    v9 = *(_QWORD *)(a2 - 48) & 0xFFFFFFFFFFFFFFF9LL;
    goto LABEL_24;
  }
  if ( (unsigned int)result > 0x1F )
  {
    if ( *(_BYTE *)(a2 + 16) != 78 )
    {
      if ( *(_BYTE *)(a2 + 16) == 79 )
      {
        v7 = *(_QWORD *)(a2 - 48);
        v8 = *(_QWORD *)(a2 - 24);
        v33 = a2 & 0xFFFFFFFFFFFFFFF9LL;
        v9 = v7 & 0xFFFFFFFFFFFFFFF9LL;
        v10 = v8 & 0xFFFFFFFFFFFFFFF9LL;
LABEL_10:
        sub_183C910((__int64)v37, a4, v10);
        sub_183C910((__int64)v34, a4, v9);
LABEL_11:
        sub_183EA00((const char *)&v40, a1, (__int64)v34, (__int64)v37);
        result = (__int64)sub_183BD80(v4, (__int64 *)&v33);
        v11 = *(_QWORD *)(result + 16);
        v12 = *(_QWORD *)(result + 32);
        *(_DWORD *)(result + 8) = v40;
        *(_QWORD *)(result + 16) = v41;
        *(_QWORD *)(result + 24) = v42;
        *(_QWORD *)(result + 32) = v43;
        v41 = 0;
        v42 = 0;
        v43 = 0;
        if ( v11 )
        {
          result = j_j___libc_free_0(v11, v12 - v11);
          if ( v41 )
            result = j_j___libc_free_0(v41, v43 - v41);
        }
        if ( v35 )
          result = j_j___libc_free_0(v35, v36 - v35);
        if ( v38 )
          return j_j___libc_free_0(v38, v39 - v38);
        return result;
      }
      goto LABEL_34;
    }
    v32 = a2 | 4;
    return (__int64)sub_183F3B0(a1, v32, a3, a4);
  }
  switch ( *(_BYTE *)(a2 + 16) )
  {
    case 0x1D:
      v32 = a2 & 0xFFFFFFFFFFFFFFFBLL;
      return (__int64)sub_183F3B0(a1, v32, a3, a4);
    case 0x36:
      a3 = *(_QWORD *)(a2 - 24);
      v33 = a2 & 0xFFFFFFFFFFFFFFF9LL;
      if ( *(_BYTE *)(a3 + 16) == 3 )
      {
        sub_183C910((__int64)v37, a4, a3 & 0xFFFFFFFFFFFFFFF9LL | 4);
        sub_183C910((__int64)v34, a4, v33);
        goto LABEL_11;
      }
      v14 = *(_BYTE **)(a1 + 56);
      v15 = *(_BYTE **)(a1 + 48);
      v16 = *(_DWORD *)(a1 + 40);
      v17 = v14 - v15;
      if ( v14 == v15 )
      {
        v20 = 0;
        v19 = 0;
LABEL_28:
        v21 = &v19[v17];
        if ( v15 != v14 )
          memmove(v19, v15, v20);
        result = (__int64)sub_183BD80(v4, (__int64 *)&v33);
        v22 = *(_QWORD *)(result + 16);
        v23 = *(_QWORD *)(result + 32);
        *(_DWORD *)(result + 8) = v16;
        *(_QWORD *)(result + 16) = v19;
        *(_QWORD *)(result + 24) = &v19[v20];
        *(_QWORD *)(result + 32) = v21;
        if ( !v22 )
          return result;
        return j_j___libc_free_0(v22, v23 - v22);
      }
      if ( v17 <= 0x7FFFFFFFFFFFFFF8LL )
      {
        v18 = sub_22077B0(v17);
        v15 = *(_BYTE **)(a1 + 48);
        v19 = (char *)v18;
        v14 = *(_BYTE **)(a1 + 56);
        v20 = v14 - v15;
        goto LABEL_28;
      }
LABEL_46:
      sub_4261EA(a1, v15, a3);
    case 0x19:
      v13 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
      result = **(_QWORD **)(*(_QWORD *)(v13 + 24) + 16LL);
      if ( !*(_BYTE *)(result + 8) )
        return result;
      v9 = 0;
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
        v9 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) & 0xFFFFFFFFFFFFFFF9LL;
      v10 = v13 & 0xFFFFFFFFFFFFFFF9LL | 2;
LABEL_24:
      v33 = v10;
      goto LABEL_10;
  }
LABEL_34:
  v25 = *(_BYTE **)(a1 + 56);
  v26 = *(_DWORD *)(a1 + 40);
  v40 = a2 & 0xFFFFFFFFFFFFFFF9LL;
  v15 = *(_BYTE **)(a1 + 48);
  v27 = v25 - v15;
  if ( v25 == v15 )
  {
    v30 = 0;
    v29 = 0;
  }
  else
  {
    if ( v27 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_46;
    v28 = sub_22077B0(v27);
    v15 = *(_BYTE **)(a1 + 48);
    v29 = (char *)v28;
    v25 = *(_BYTE **)(a1 + 56);
    v30 = v25 - v15;
  }
  v31 = &v29[v27];
  if ( v25 != v15 )
    memmove(v29, v15, v30);
  result = (__int64)sub_183BD80(v4, (__int64 *)&v40);
  v22 = *(_QWORD *)(result + 16);
  v23 = *(_QWORD *)(result + 32);
  *(_DWORD *)(result + 8) = v26;
  *(_QWORD *)(result + 16) = v29;
  *(_QWORD *)(result + 24) = &v29[v30];
  *(_QWORD *)(result + 32) = v31;
  if ( v22 )
    return j_j___libc_free_0(v22, v23 - v22);
  return result;
}
