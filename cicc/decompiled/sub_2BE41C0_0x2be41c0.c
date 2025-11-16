// Function: sub_2BE41C0
// Address: 0x2be41c0
//
__int64 __fastcall sub_2BE41C0(char *a1)
{
  __int64 v2; // r15
  char v3; // r13
  unsigned int v4; // eax
  unsigned int v5; // r12d
  void *v7; // r14
  _BYTE *v8; // r13
  __int64 v9; // rax
  __int64 v10; // r8
  _QWORD *v11; // rax
  __int64 *v12; // r14
  void *v13; // r13
  __int64 v14; // r15
  unsigned __int64 v15; // r12
  size_t v16; // rdx
  int v17; // eax
  size_t v18; // rcx
  size_t v19; // rdx
  signed __int64 v20; // rax
  unsigned int v21; // eax
  __int64 v22; // r13
  __int64 v23; // r14
  __int64 v24; // rax
  _QWORD *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // r14
  __int64 v29; // [rsp+8h] [rbp-B8h]
  size_t v30; // [rsp+20h] [rbp-A0h]
  __int64 v31; // [rsp+28h] [rbp-98h]
  __int64 v32; // [rsp+28h] [rbp-98h]
  __int64 v33; // [rsp+28h] [rbp-98h]
  void *s2[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v35; // [rsp+40h] [rbp-80h] BYREF
  void *src[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v37[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD *v38; // [rsp+70h] [rbp-50h] BYREF
  void *v39; // [rsp+78h] [rbp-48h]
  _QWORD v40[8]; // [rsp+80h] [rbp-40h] BYREF

  v2 = *(_QWORD *)a1;
  v3 = a1[8];
  LOBYTE(v38) = v3;
  LOBYTE(v4) = sub_2BE37E0(*(char **)v2, *(_BYTE **)(v2 + 8), (char *)&v38);
  v5 = v4;
  if ( (_BYTE)v4 )
    return v5;
  src[0] = v37;
  sub_2240A50((__int64 *)src, 1u, v3);
  v7 = src[1];
  v8 = src[0];
  v9 = sub_221F880(*(_QWORD **)(v2 + 104), 1);
  s2[0] = v7;
  v10 = v9;
  v38 = v40;
  if ( (unsigned __int64)v7 > 0xF )
  {
    v32 = v9;
    v24 = sub_22409D0((__int64)&v38, (unsigned __int64 *)s2, 0);
    v10 = v32;
    v38 = (_QWORD *)v24;
    v25 = (_QWORD *)v24;
    v40[0] = s2[0];
  }
  else
  {
    if ( v7 == (void *)1 )
    {
      LOBYTE(v40[0]) = *v8;
      v11 = v40;
      goto LABEL_6;
    }
    if ( !v7 )
    {
      v11 = v40;
      goto LABEL_6;
    }
    v25 = v40;
  }
  v33 = v10;
  memcpy(v25, v8, (size_t)v7);
  v7 = s2[0];
  v11 = v38;
  v10 = v33;
LABEL_6:
  v39 = v7;
  *((_BYTE *)v7 + (_QWORD)v11) = 0;
  (*(void (__fastcall **)(void **, __int64, _QWORD *, unsigned __int64))(*(_QWORD *)v10 + 24LL))(
    s2,
    v10,
    v38,
    (unsigned __int64)v39 + (_QWORD)v38);
  if ( v38 != v40 )
    j_j___libc_free_0((unsigned __int64)v38);
  if ( src[0] != v37 )
    j_j___libc_free_0((unsigned __int64)src[0]);
  v29 = *(_QWORD *)a1;
  v31 = *(_QWORD *)(*(_QWORD *)a1 + 56LL);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 48LL) == v31 )
  {
LABEL_33:
    LOBYTE(v21) = sub_2BDBFE0(*(_QWORD **)(v29 + 112), (unsigned int)a1[8], *(_WORD *)(v29 + 96), *(_BYTE *)(v29 + 98));
    v5 = v21;
    if ( !(_BYTE)v21 )
    {
      v22 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      sub_2BE40B0((__int64)&v38, *(_QWORD **)(*(_QWORD *)a1 + 112LL), a1 + 8, (__int64)(a1 + 9));
      v23 = sub_2BDD0F0(*(_QWORD *)(*(_QWORD *)a1 + 24LL), *(_QWORD *)(*(_QWORD *)a1 + 32LL), (__int64)&v38);
      if ( v38 != v40 )
        j_j___libc_free_0((unsigned __int64)v38);
      if ( v22 != v23 )
      {
LABEL_37:
        v12 = (__int64 *)s2[0];
        v5 = 1;
        goto LABEL_28;
      }
      v26 = *(_QWORD *)a1;
      v27 = *(_QWORD *)(*(_QWORD *)a1 + 72LL);
      v28 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
      if ( v28 != v27 )
      {
        while ( sub_2BDBFE0(*(_QWORD **)(v26 + 112), (unsigned int)a1[8], *(_WORD *)v27, *(_BYTE *)(v27 + 2)) )
        {
          v27 += 4;
          if ( v28 == v27 )
            goto LABEL_45;
          v26 = *(_QWORD *)a1;
        }
        goto LABEL_37;
      }
    }
LABEL_45:
    v12 = (__int64 *)s2[0];
    goto LABEL_28;
  }
  v12 = (__int64 *)s2[0];
  v13 = s2[1];
  v14 = *(_QWORD *)(*(_QWORD *)a1 + 48LL);
  while ( 1 )
  {
    v15 = *(_QWORD *)(v14 + 8);
    v16 = (size_t)v13;
    if ( v15 <= (unsigned __int64)v13 )
      v16 = *(_QWORD *)(v14 + 8);
    if ( v16 )
    {
      v17 = memcmp(*(const void **)v14, v12, v16);
      if ( v17 )
        goto LABEL_19;
    }
    if ( (__int64)(v15 - (_QWORD)v13) >= 0x80000000LL )
      goto LABEL_32;
    if ( (__int64)(v15 - (_QWORD)v13) > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
      v17 = v15 - (_DWORD)v13;
LABEL_19:
      if ( v17 > 0 )
        goto LABEL_32;
    }
    v18 = *(_QWORD *)(v14 + 40);
    v19 = v18;
    if ( (unsigned __int64)v13 <= v18 )
      v19 = (size_t)v13;
    if ( v19 )
    {
      v30 = *(_QWORD *)(v14 + 40);
      LODWORD(v20) = memcmp(v12, *(const void **)(v14 + 32), v19);
      v18 = v30;
      if ( (_DWORD)v20 )
        goto LABEL_26;
    }
    v20 = (signed __int64)v13 - v18;
    if ( (__int64)((__int64)v13 - v18) < 0x80000000LL )
      break;
LABEL_32:
    v14 += 64;
    if ( v31 == v14 )
      goto LABEL_33;
  }
  if ( v20 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
    goto LABEL_27;
LABEL_26:
  if ( (int)v20 > 0 )
    goto LABEL_32;
LABEL_27:
  v5 = 1;
LABEL_28:
  if ( v12 != &v35 )
    j_j___libc_free_0((unsigned __int64)v12);
  return v5;
}
