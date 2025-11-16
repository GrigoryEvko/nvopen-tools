// Function: sub_2B315C0
// Address: 0x2b315c0
//
void __fastcall sub_2B315C0(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, _BYTE *),
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // r12
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 v14; // r14
  __int64 *v15; // r12
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned int v18; // edx
  _BYTE *v19; // rbx
  char v20; // al
  __int64 v21; // r8
  bool v22; // zf
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned int v25; // r14d
  size_t v26; // rdx
  __int64 v27; // r12
  _BYTE *v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r12
  const void *v33; // [rsp+0h] [rbp-F0h]
  unsigned int *v36; // [rsp+30h] [rbp-C0h]
  unsigned int v37; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v38[2]; // [rsp+40h] [rbp-B0h] BYREF
  _BYTE v39[48]; // [rsp+50h] [rbp-A0h] BYREF
  void *v40; // [rsp+80h] [rbp-70h] BYREF
  __int64 v41; // [rsp+88h] [rbp-68h]
  _BYTE s[96]; // [rsp+90h] [rbp-60h] BYREF

  v6 = a4;
  v8 = a6;
  v10 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a4 + 12) < (unsigned int)v10 )
  {
    *(_DWORD *)(a4 + 8) = 0;
    sub_C8D5F0(a4, (const void *)(a4 + 16), v10, 4u, a5, a6);
    memset(*(void **)v6, 255, 4 * v10);
    *(_DWORD *)(v6 + 8) = v10;
  }
  else
  {
    v11 = *(unsigned int *)(a4 + 8);
    v12 = v10;
    if ( v10 > v11 )
      v12 = *(unsigned int *)(a4 + 8);
    if ( v12 )
    {
      memset(*(void **)a4, 255, 4 * v12);
      v11 = *(unsigned int *)(v6 + 8);
    }
    if ( v10 > v11 && v10 != v11 )
    {
      a4 = *(_QWORD *)v6;
      v26 = 4 * (v10 - v11);
      if ( v26 )
        memset((void *)(*(_QWORD *)v6 + 4 * v11), 255, v26);
    }
    *(_DWORD *)(v6 + 8) = v10;
  }
  v13 = *(_DWORD *)(a1 + 152);
  v38[0] = (unsigned __int64)v39;
  v38[1] = 0xC00000000LL;
  if ( v13 )
    sub_2B0FC00(*(_QWORD *)(a1 + 144), v13, (__int64)v38, a4, a5, a6);
  v14 = 0;
  v33 = (const void *)(a5 + 16);
  if ( (_DWORD)v10 )
  {
    a6 = v8;
    v15 = (__int64 *)v6;
    v16 = a5;
    while ( 1 )
    {
      v18 = v14;
      if ( *(_DWORD *)(a1 + 152) )
        v18 = *(_DWORD *)(v38[0] + 4 * v14);
      a4 = *(_QWORD *)a1;
      v37 = v18;
      v19 = *(_BYTE **)(*(_QWORD *)a1 + 8LL * v18);
      if ( *v19 == 13 )
        goto LABEL_16;
      v36 = (unsigned int *)a6;
      v20 = a2(a3, v19);
      a6 = (__int64)v36;
      v22 = v20 == 0;
      v23 = *v15;
      if ( !v22 )
        break;
      *(_DWORD *)(v23 + 4 * v14) = v37;
      if ( v16 )
      {
        v24 = *(unsigned int *)(v16 + 8);
        a4 = *(unsigned int *)(v16 + 12);
        if ( v24 + 1 > a4 )
        {
          sub_C8D5F0(v16, v33, v24 + 1, 8u, v21, (__int64)v36);
          v24 = *(unsigned int *)(v16 + 8);
          a6 = (__int64)v36;
        }
        ++v14;
        *(_QWORD *)(*(_QWORD *)v16 + 8 * v24) = v19;
        ++*(_DWORD *)(v16 + 8);
        if ( v10 == v14 )
        {
LABEL_25:
          v6 = (__int64)v15;
          goto LABEL_26;
        }
      }
      else
      {
LABEL_16:
        if ( v10 == ++v14 )
          goto LABEL_25;
      }
    }
    *(_DWORD *)(v23 + 4 * v14) = v10 + v37;
    if ( v36 )
    {
      v17 = v36[2];
      a4 = v36[3];
      if ( v17 + 1 > a4 )
      {
        sub_C8D5F0((__int64)v36, v36 + 4, v17 + 1, 8u, v21, (__int64)v36);
        a6 = (__int64)v36;
        v17 = v36[2];
      }
      *(_QWORD *)(*(_QWORD *)a6 + 8 * v17) = v19;
      ++*(_DWORD *)(a6 + 8);
    }
    goto LABEL_16;
  }
LABEL_26:
  v25 = *(_DWORD *)(a1 + 120);
  if ( v25 )
  {
    v40 = s;
    v27 = 4LL * v25;
    v41 = 0xC00000000LL;
    if ( v25 > 0xC )
    {
      sub_C8D5F0((__int64)&v40, s, v25, 4u, (__int64)&v40, a6);
      memset(v40, 255, 4LL * v25);
      v32 = *(unsigned int *)(a1 + 120);
      LODWORD(v41) = v25;
      v28 = v40;
      v27 = 4 * v32;
    }
    else
    {
      if ( v27 )
        memset(s, 255, 4LL * v25);
      LODWORD(v41) = v25;
      v28 = s;
    }
    v29 = *(_QWORD *)(a1 + 112);
    v30 = 0;
    if ( v27 )
    {
      do
      {
        v31 = *(int *)(v29 + v30);
        if ( (_DWORD)v31 != -1 )
        {
          a4 = *(_QWORD *)v6;
          LODWORD(v31) = *(_DWORD *)(*(_QWORD *)v6 + 4 * v31);
        }
        *(_DWORD *)&v28[v30] = v31;
        v30 += 4;
      }
      while ( v27 != v30 );
    }
    sub_2B310D0(v6, (__int64)&v40, v30, a4, (__int64)&v40, a6);
    if ( v40 != s )
      _libc_free((unsigned __int64)v40);
  }
  if ( (_BYTE *)v38[0] != v39 )
    _libc_free(v38[0]);
}
