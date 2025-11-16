// Function: sub_EE2370
// Address: 0xee2370
//
unsigned __int64 *__fastcall sub_EE2370(unsigned __int64 *a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  unsigned __int64 v6; // r14
  __int64 v10; // rdx
  __int64 v11; // rax
  int v12; // ecx
  __int64 v13; // rdi
  int v14; // ecx
  __int64 v15; // rdx
  char *v16; // r10
  char *v17; // r9
  __int64 v18; // r8
  __int64 *v19; // rsi
  __int64 v20; // r11
  __int64 v21; // rax
  char *v22; // rdi
  char *v23; // r11
  char *v24; // rsi
  unsigned __int64 v25; // rdx
  __int64 *v26; // rsi
  unsigned __int64 v27; // rax
  unsigned __int64 v29; // rax
  int v30; // esi
  char *v31; // [rsp+10h] [rbp-1B0h]
  __int64 v32; // [rsp+18h] [rbp-1A8h]
  __int64 v33; // [rsp+18h] [rbp-1A8h]
  char *v34; // [rsp+20h] [rbp-1A0h]
  char *v35; // [rsp+20h] [rbp-1A0h]
  char *v37; // [rsp+38h] [rbp-188h]
  char *v38; // [rsp+38h] [rbp-188h]
  __int64 v39; // [rsp+40h] [rbp-180h]
  int v40; // [rsp+40h] [rbp-180h]
  __int64 v42; // [rsp+58h] [rbp-168h] BYREF
  __int64 v43; // [rsp+60h] [rbp-160h] BYREF
  unsigned __int64 v44; // [rsp+68h] [rbp-158h] BYREF
  _BYTE *v45; // [rsp+70h] [rbp-150h] BYREF
  __int64 v46; // [rsp+78h] [rbp-148h]
  __int64 v47; // [rsp+80h] [rbp-140h]
  _BYTE v48[312]; // [rsp+88h] [rbp-138h] BYREF

  v5 = a3;
  v6 = a4;
  v37 = (char *)sub_EE20B0(a3, a4);
  v39 = v10;
  v11 = sub_EF7600(a2 + 16, v37, v10);
  if ( !v11 )
    goto LABEL_14;
  v12 = *(_DWORD *)(a2 + 48);
  v13 = *(_QWORD *)(a2 + 32);
  if ( !v12 )
    goto LABEL_14;
  v14 = v12 - 1;
  v15 = v39;
  v16 = v37;
  v17 = (char *)a3;
  v18 = v14 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v11) >> 31) ^ (484763065 * (_DWORD)v11));
  v19 = (__int64 *)(v13 + 24 * v18);
  v20 = *v19;
  if ( v11 != *v19 )
  {
    v30 = 1;
    while ( v20 != -1 )
    {
      v18 = v14 & (unsigned int)(v30 + v18);
      v40 = v30 + 1;
      v19 = (__int64 *)(v13 + 24LL * (unsigned int)v18);
      v20 = *v19;
      if ( v11 == *v19 )
        goto LABEL_4;
      v30 = v40;
    }
    goto LABEL_14;
  }
LABEL_4:
  v21 = v19[2];
  if ( !v21 )
  {
LABEL_14:
    (*(void (__fastcall **)(unsigned __int64 *, _QWORD, __int64, unsigned __int64, __int64))(**(_QWORD **)(a2 + 56)
                                                                                           + 24LL))(
      a1,
      *(_QWORD *)(a2 + 56),
      v5,
      v6,
      a5);
    return a1;
  }
  v22 = &v37[v15];
  v23 = (char *)v19[1];
  v38 = &v37[v15];
  if ( v16 == (char *)a3 && (char *)(a3 + a4) == v22 )
  {
    v6 = v19[2];
    v5 = v19[1];
    goto LABEL_14;
  }
  v24 = v48;
  v46 = 0;
  v25 = a4 - v15 + v21;
  v45 = v48;
  v47 = 256;
  if ( v25 > 0x100 )
  {
    v31 = v16;
    v33 = v21;
    v35 = v23;
    sub_C8D290((__int64)&v45, v48, v25, 1u, v18, a3);
    v17 = (char *)a3;
    v16 = v31;
    v21 = v33;
    v23 = v35;
    v24 = &v45[v46];
  }
  v32 = v21;
  v34 = v23;
  sub_ED6EC0(&v45, v24, v17, v16);
  sub_ED6EC0(&v45, &v45[v46], v34, &v34[v32]);
  sub_ED6EC0(&v45, &v45[v46], v38, (char *)(a3 + a4));
  v26 = *(__int64 **)(a2 + 56);
  (*(void (__fastcall **)(__int64 *, __int64 *, _BYTE *, __int64, __int64))(*v26 + 24))(&v42, v26, v45, v46, a5);
  v27 = v42 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v42 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v42 = 0;
    v26 = (__int64 *)&v44;
    v44 = v27 | 1;
    sub_EE21A0((unsigned __int64 *)&v43, (__int64 *)&v44);
    sub_9C66B0((__int64 *)&v44);
    v29 = v43 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v43 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      v43 = 0;
      sub_9C66B0(&v43);
      sub_9C66B0(&v42);
      if ( v45 != v48 )
        _libc_free(v45, &v44);
      goto LABEL_14;
    }
    v43 = 0;
    *a1 = v29 | 1;
    sub_9C66B0(&v43);
  }
  else
  {
    *a1 = 1;
    v42 = 0;
  }
  sub_9C66B0(&v42);
  if ( v45 != v48 )
    _libc_free(v45, v26);
  return a1;
}
