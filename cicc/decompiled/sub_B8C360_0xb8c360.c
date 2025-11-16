// Function: sub_B8C360
// Address: 0xb8c360
//
__int64 __fastcall sub_B8C360(_QWORD *a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r13
  const char *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rdx
  _QWORD *v19; // r9
  _QWORD *v20; // r8
  __int64 *v21; // r15
  __int64 v22; // rsi
  __int64 *v23; // r12
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rdx
  _BYTE *v28; // rsi
  __int64 v29; // r12
  _QWORD *v31; // rax
  __int64 v32; // r12
  _QWORD *v33; // rdx
  char *v34; // rcx
  __int64 v35; // rdx
  _QWORD *v36; // rax
  _QWORD *v37; // [rsp+0h] [rbp-C0h]
  _QWORD *v38; // [rsp+8h] [rbp-B8h]
  __int64 v39; // [rsp+8h] [rbp-B8h]
  __int64 v41; // [rsp+18h] [rbp-A8h]
  void *base; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v43; // [rsp+28h] [rbp-98h]
  _BYTE v44[16]; // [rsp+30h] [rbp-90h] BYREF
  _BYTE *v45; // [rsp+40h] [rbp-80h] BYREF
  __int64 v46; // [rsp+48h] [rbp-78h]
  _BYTE v47[112]; // [rsp+50h] [rbp-70h] BYREF

  v6 = sub_BCB2E0(*a1);
  v45 = v47;
  v7 = v6;
  v46 = 0x800000000LL;
  if ( a3 )
  {
    v8 = "synthetic_function_entry_count";
    v9 = 30;
  }
  else
  {
    v8 = "function_entry_count";
    v9 = 20;
  }
  v10 = sub_B8C130(a1, (__int64)v8, v9);
  v11 = (unsigned int)v46;
  if ( (unsigned __int64)(unsigned int)v46 + 1 > HIDWORD(v46) )
  {
    v41 = v10;
    sub_C8D5F0(&v45, v47, (unsigned int)v46 + 1LL, 8);
    v11 = (unsigned int)v46;
    v10 = v41;
  }
  *(_QWORD *)&v45[8 * v11] = v10;
  LODWORD(v46) = v46 + 1;
  v12 = sub_AD64C0(v7, a2, 0);
  v15 = sub_B8C140((__int64)a1, v12, v13, v14);
  v16 = (unsigned int)v46;
  v17 = (unsigned int)v46 + 1LL;
  if ( v17 > HIDWORD(v46) )
  {
    sub_C8D5F0(&v45, v47, v17, 8);
    v16 = (unsigned int)v46;
  }
  *(_QWORD *)&v45[8 * v16] = v15;
  v18 = (unsigned int)(v46 + 1);
  LODWORD(v46) = v46 + 1;
  if ( a4 )
  {
    v19 = *(_QWORD **)(a4 + 8);
    v20 = &v19[*(unsigned int *)(a4 + 24)];
    if ( !*(_DWORD *)(a4 + 16) || v19 == v20 )
    {
LABEL_9:
      HIDWORD(v43) = 2;
      base = v44;
    }
    else
    {
      while ( *v19 > 0xFFFFFFFFFFFFFFFDLL )
      {
        if ( v20 == ++v19 )
          goto LABEL_9;
      }
      base = v44;
      v43 = 0x200000000LL;
      if ( v20 != v19 )
      {
        v31 = v19;
        v32 = 0;
        while ( 1 )
        {
          v33 = v31 + 1;
          if ( v20 == v31 + 1 )
            break;
          while ( 1 )
          {
            v31 = v33;
            if ( *v33 <= 0xFFFFFFFFFFFFFFFDLL )
              break;
            if ( v20 == ++v33 )
              goto LABEL_32;
          }
          ++v32;
          if ( v20 == v33 )
            goto LABEL_33;
        }
LABEL_32:
        ++v32;
LABEL_33:
        v34 = v44;
        if ( v32 > 2 )
        {
          v37 = v19;
          v38 = v20;
          sub_C8D5F0(&base, v44, v32, 8);
          v20 = v38;
          v19 = v37;
          v34 = (char *)base + 8 * (unsigned int)v43;
        }
        v35 = *v19;
        do
        {
          v36 = v19 + 1;
          *(_QWORD *)v34 = v35;
          v34 += 8;
          if ( v20 == v19 + 1 )
            break;
          while ( 1 )
          {
            v35 = *v36;
            v19 = v36;
            if ( *v36 <= 0xFFFFFFFFFFFFFFFDLL )
              break;
            if ( v20 == ++v36 )
              goto LABEL_39;
          }
        }
        while ( v20 != v36 );
LABEL_39:
        v21 = (__int64 *)base;
        LODWORD(v43) = v43 + v32;
        v22 = 8LL * (unsigned int)v43;
        if ( (unsigned int)v43 > 1uLL )
        {
          qsort(base, v22 >> 3, 8u, (__compar_fn_t)sub_A15280);
          v21 = (__int64 *)base;
          v22 = 8LL * (unsigned int)v43;
        }
        goto LABEL_11;
      }
    }
    LODWORD(v43) = 0;
    v21 = (__int64 *)v44;
    v22 = 0;
LABEL_11:
    v23 = (__int64 *)((char *)v21 + v22);
    if ( v21 != (__int64 *)((char *)v21 + v22) )
    {
      do
      {
        v22 = sub_AD64C0(v7, *v21, 0);
        v26 = sub_B8C140((__int64)a1, v22, v24, v25);
        v27 = (unsigned int)v46;
        if ( (unsigned __int64)(unsigned int)v46 + 1 > HIDWORD(v46) )
        {
          v22 = (__int64)v47;
          v39 = v26;
          sub_C8D5F0(&v45, v47, (unsigned int)v46 + 1LL, 8);
          v27 = (unsigned int)v46;
          v26 = v39;
        }
        ++v21;
        *(_QWORD *)&v45[8 * v27] = v26;
        LODWORD(v46) = v46 + 1;
      }
      while ( v23 != v21 );
      v23 = (__int64 *)base;
    }
    if ( v23 != (__int64 *)v44 )
      _libc_free(v23, v22);
    v18 = (unsigned int)v46;
  }
  v28 = v45;
  v29 = sub_B9C770(*a1, v45, v18, 0, 1);
  if ( v45 != v47 )
    _libc_free(v45, v28);
  return v29;
}
