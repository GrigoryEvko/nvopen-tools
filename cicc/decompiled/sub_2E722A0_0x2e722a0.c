// Function: sub_2E722A0
// Address: 0x2e722a0
//
__int64 __fastcall sub_2E722A0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 *v9; // r14
  __int64 v10; // r12
  __int64 *v11; // r13
  __int64 v12; // r15
  unsigned int v13; // eax
  void *v14; // rax
  _QWORD *v15; // rdi
  _BYTE *v16; // rax
  __int64 *v17; // rdi
  __int64 result; // rax
  unsigned int v19; // r9d
  const void *v20; // r10
  char *v21; // rdi
  size_t v22; // r11
  __int64 v23; // r9
  char *v24; // rcx
  unsigned __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rsi
  void *v29; // rax
  _QWORD *v30; // rdi
  _BYTE *v31; // rax
  __int64 *v32; // rdi
  char *i; // r9
  __int64 v34; // rdx
  __int64 v35; // rsi
  char *v36; // rax
  char *v37; // rdi
  unsigned __int64 v38; // [rsp-C8h] [rbp-C8h]
  __int64 v39; // [rsp-C8h] [rbp-C8h]
  char *v40; // [rsp-C0h] [rbp-C0h]
  const void *v41; // [rsp-C0h] [rbp-C0h]
  unsigned int v42; // [rsp-B8h] [rbp-B8h]
  char *v43; // [rsp-B8h] [rbp-B8h]
  unsigned int v44; // [rsp-B8h] [rbp-B8h]
  __int64 v45[4]; // [rsp-A8h] [rbp-A8h] BYREF
  char *v46; // [rsp-88h] [rbp-88h] BYREF
  __int64 v47; // [rsp-80h] [rbp-80h]
  _BYTE v48[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( !*((_BYTE *)a1 + 112) || !a1[13] )
    return 1;
  v5 = **a1;
  if ( v5 )
  {
    v6 = (unsigned int)(*(_DWORD *)(v5 + 24) + 1);
    v7 = *(_DWORD *)(v5 + 24) + 1;
  }
  else
  {
    v6 = 0;
    v7 = 0;
  }
  v8 = *((unsigned int *)a1 + 8);
  if ( (unsigned int)v8 <= v7 )
    BUG();
  v9 = a1[3];
  v10 = v9[v6];
  if ( *(_DWORD *)(v10 + 72) )
  {
    v29 = sub_CB72A0();
    sub_904010((__int64)v29, "DFSIn number for the tree root is not:\n\t");
    sub_2E6C8E0(v10);
    v30 = sub_CB72A0();
    v31 = (_BYTE *)v30[4];
    if ( (unsigned __int64)v31 >= v30[3] )
    {
      sub_CB5D20((__int64)v30, 10);
    }
    else
    {
      v30[4] = v31 + 1;
      *v31 = 10;
    }
    v32 = (__int64 *)sub_CB72A0();
    result = 0;
    if ( v32[2] != v32[4] )
    {
      sub_CB5AE0(v32);
      return 0;
    }
    return result;
  }
  v11 = &v9[v8];
  if ( v11 == v9 )
    return 1;
  while ( 1 )
  {
    v12 = *v9;
    if ( !*v9 )
      goto LABEL_30;
    v13 = *(_DWORD *)(v12 + 32);
    if ( v13 )
      break;
    if ( *(_DWORD *)(v12 + 72) + 1 != *(_DWORD *)(v12 + 76) )
    {
      v14 = sub_CB72A0();
      sub_904010((__int64)v14, "Tree leaf should have DFSOut = DFSIn + 1:\n\t");
      sub_2E6C8E0(v12);
      v15 = sub_CB72A0();
      v16 = (_BYTE *)v15[4];
      if ( (unsigned __int64)v16 >= v15[3] )
      {
        sub_CB5D20((__int64)v15, 10);
      }
      else
      {
        v15[4] = v16 + 1;
        *v16 = 10;
      }
      v17 = (__int64 *)sub_CB72A0();
      if ( v17[4] != v17[2] )
        sub_CB5AE0(v17);
      return 0;
    }
LABEL_30:
    if ( v11 == ++v9 )
      return 1;
  }
  v19 = *(_DWORD *)(v12 + 32);
  v47 = 0x800000000LL;
  v20 = *(const void **)(v12 + 24);
  v21 = v48;
  v22 = 8LL * v13;
  v46 = v48;
  if ( v13 > 8uLL )
  {
    v39 = 8LL * v13;
    v41 = v20;
    v44 = v13;
    sub_C8D5F0((__int64)&v46, v48, v13, 8u, a5, v13);
    v19 = v44;
    v20 = v41;
    v22 = v39;
    v21 = &v46[8 * (unsigned int)v47];
  }
  v42 = v19;
  memcpy(v21, v20, v22);
  LODWORD(v47) = v47 + v42;
  v23 = 8LL * (unsigned int)v47;
  v24 = &v46[v23];
  if ( v46 != &v46[v23] )
  {
    v38 = 8LL * (unsigned int)v47;
    v40 = &v46[v23];
    _BitScanReverse64(&v25, v23 >> 3);
    v43 = v46;
    sub_2E720E0(v46, (__int64 *)&v46[v23], 2LL * (int)(63 - (v25 ^ 0x3F)));
    if ( v38 > 0x80 )
    {
      sub_2E6C260(v43, v43 + 128);
      for ( i = v43 + 128; v40 != i; *(_QWORD *)v37 = v35 )
      {
        v34 = *((_QWORD *)i - 1);
        v35 = *(_QWORD *)i;
        v36 = i - 8;
        if ( *(_DWORD *)(*(_QWORD *)i + 72LL) >= *(_DWORD *)(v34 + 72) )
        {
          v37 = i;
        }
        else
        {
          do
          {
            *((_QWORD *)v36 + 1) = v34;
            v37 = v36;
            v34 = *((_QWORD *)v36 - 1);
            v36 -= 8;
            a5 = *(unsigned int *)(v34 + 72);
          }
          while ( *(_DWORD *)(v35 + 72) < (unsigned int)a5 );
        }
        i += 8;
      }
    }
    else
    {
      sub_2E6C260(v43, v40);
    }
    v24 = v46;
  }
  v45[0] = v12;
  v45[2] = 0;
  v45[1] = (__int64)&v46;
  v26 = *(_QWORD *)v24;
  if ( *(_DWORD *)(v12 + 72) + 1 == *(_DWORD *)(*(_QWORD *)v24 + 72LL) )
  {
    v26 = *(_QWORD *)&v24[8 * (unsigned int)v47 - 8];
    if ( *(_DWORD *)(v26 + 76) + 1 == *(_DWORD *)(v12 + 76) )
    {
      v27 = 0;
      while ( v27 != (unsigned int)v47 - 1LL )
      {
        v28 = *(_QWORD *)&v24[8 * v27++];
        if ( *(_DWORD *)(v28 + 76) + 1 != *(_DWORD *)(*(_QWORD *)&v24[8 * v27] + 72LL) )
        {
          sub_2E6DEF0(v45, v28, *(_QWORD *)&v24[8 * v27]);
          goto LABEL_39;
        }
      }
      if ( v24 != v48 )
        _libc_free((unsigned __int64)v24);
      goto LABEL_30;
    }
  }
  sub_2E6DEF0(v45, v26, 0);
LABEL_39:
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  return 0;
}
