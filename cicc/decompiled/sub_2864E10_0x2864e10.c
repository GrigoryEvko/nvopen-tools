// Function: sub_2864E10
// Address: 0x2864e10
//
__int64 __fastcall sub_2864E10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v7; // rbx
  char *v8; // rax
  _BYTE *v9; // r12
  size_t v10; // rsi
  unsigned __int64 v11; // rax
  int v12; // r15d
  __int64 v13; // rax
  int v14; // r15d
  size_t v15; // r13
  int v16; // eax
  __int64 v17; // r8
  int v18; // r9d
  unsigned int i; // ecx
  __int64 v20; // rbx
  __int64 v21; // r10
  unsigned int v22; // ecx
  unsigned int v23; // r13d
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int64 v27; // rdx
  int v28; // eax
  __int64 v29; // [rsp+8h] [rbp-B8h]
  __int64 v30; // [rsp+10h] [rbp-B0h]
  int v31; // [rsp+1Ch] [rbp-A4h]
  __int64 v32; // [rsp+20h] [rbp-A0h]
  unsigned int v33; // [rsp+20h] [rbp-A0h]
  __int64 v34; // [rsp+28h] [rbp-98h]
  void *base; // [rsp+30h] [rbp-90h] BYREF
  __int64 v36; // [rsp+38h] [rbp-88h]
  _BYTE s1[48]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v38; // [rsp+70h] [rbp-50h]

  v36 = 0x400000000LL;
  v6 = *(_DWORD *)(a2 + 48);
  base = s1;
  if ( !v6 )
  {
    v7 = *(_QWORD *)(a2 + 88);
    v8 = s1;
    v9 = s1;
    if ( !v7 )
      goto LABEL_6;
    goto LABEL_3;
  }
  sub_2850210((__int64)&base, a2 + 40, a3, a4, a5, a6);
  v7 = *(_QWORD *)(a2 + 88);
  if ( v7 )
  {
    v27 = (unsigned int)v36 + 1LL;
    if ( v27 > HIDWORD(v36) )
      sub_C8D5F0((__int64)&base, s1, v27, 8u, v25, v26);
    v8 = (char *)base + 8 * (unsigned int)v36;
LABEL_3:
    *(_QWORD *)v8 = v7;
    v9 = base;
    LODWORD(v36) = v36 + 1;
    v10 = (unsigned int)v36;
    v11 = 8LL * (unsigned int)v36;
    goto LABEL_4;
  }
  v10 = (unsigned int)v36;
  v9 = base;
  v11 = 8LL * (unsigned int)v36;
LABEL_4:
  if ( v11 > 8 )
  {
    qsort(v9, v10, 8u, (__compar_fn_t)sub_284F380);
    v9 = base;
  }
LABEL_6:
  v12 = *(_DWORD *)(a1 + 24);
  if ( v12 )
  {
    v13 = *(_QWORD *)(a1 + 8);
    v38 = -1;
    v14 = v12 - 1;
    v15 = 8LL * (unsigned int)v36;
    v32 = (unsigned int)v36;
    v34 = v13;
    v16 = sub_2862070(v9, (__int64)&v9[v15]);
    v17 = v32;
    v18 = 1;
    for ( i = v14 & v16; ; i = v14 & v22 )
    {
      v20 = v34 + 48LL * i;
      v21 = *(unsigned int *)(v20 + 8);
      if ( v17 == v21 )
      {
        v29 = *(unsigned int *)(v20 + 8);
        v30 = v17;
        v31 = v18;
        v33 = i;
        if ( !v15 )
          break;
        v28 = memcmp(v9, *(const void **)v20, v15);
        i = v33;
        v18 = v31;
        v17 = v30;
        v21 = v29;
        if ( !v28 )
          break;
      }
      if ( v21 == 1 && **(_QWORD **)v20 == v38 )
        goto LABEL_17;
      v22 = v18 + i;
      ++v18;
    }
    v23 = 1;
  }
  else
  {
LABEL_17:
    v23 = 0;
  }
  if ( v9 != s1 )
    _libc_free((unsigned __int64)v9);
  return v23;
}
