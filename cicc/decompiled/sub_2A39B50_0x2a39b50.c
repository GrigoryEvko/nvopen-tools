// Function: sub_2A39B50
// Address: 0x2a39b50
//
__int64 __fastcall sub_2A39B50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 **a5,
        __int64 a6,
        void (__fastcall *a7)(__int64, __int64),
        __int64 a8)
{
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r9
  __int64 *v14; // rcx
  __int64 v15; // r8
  __int64 *v16; // r14
  __int64 *v17; // rbx
  __int64 v18; // rsi
  __int64 *v19; // rax
  __int64 *v20; // r9
  __int64 *v21; // r15
  __int64 v22; // r14
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 *v28; // rax
  __int64 *v29; // rdx
  __int64 *v30; // rbx
  __int64 *v31; // r12
  __int64 v32; // rsi
  unsigned int v33; // r14d
  __int64 *v35; // rbx
  __int64 v36; // r12
  __int64 v37; // rsi
  unsigned int v39; // eax
  __int64 v41; // [rsp+30h] [rbp-C0h]
  int v42; // [rsp+30h] [rbp-C0h]
  __int64 v44; // [rsp+38h] [rbp-B8h]
  __int64 v45; // [rsp+40h] [rbp-B0h] BYREF
  __int64 *v46; // [rsp+48h] [rbp-A8h]
  __int64 v47; // [rsp+50h] [rbp-A0h]
  int v48; // [rsp+58h] [rbp-98h]
  unsigned __int8 v49; // [rsp+5Ch] [rbp-94h]
  _BYTE v50[16]; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v51; // [rsp+70h] [rbp-80h] BYREF
  __int64 v52; // [rsp+78h] [rbp-78h]
  _BYTE v53[112]; // [rsp+80h] [rbp-70h] BYREF

  v9 = a4;
  v10 = a8;
  v11 = *((unsigned int *)a5 + 2);
  if ( (_DWORD)v11 == 1 )
  {
    sub_104C6D0(a2, **a5, a4);
    v33 = v39;
    if ( (_BYTE)v39 )
    {
      a7(a8, **a5);
      return v33;
    }
    v11 = *((unsigned int *)a5 + 2);
  }
  v12 = (__int64 *)v50;
  v45 = 0;
  v13 = 1;
  v46 = (__int64 *)v50;
  v14 = *a5;
  v49 = 1;
  v47 = 2;
  v48 = 0;
  v15 = (__int64)&v14[v11];
  if ( (__int64 *)v15 != v14 )
  {
    v41 = v9;
    v16 = &v14[v11];
    v17 = v14;
    while ( 1 )
    {
      while ( 1 )
      {
        v18 = *(_QWORD *)(*v17 + 40);
        if ( (_BYTE)v13 )
          break;
LABEL_33:
        ++v17;
        sub_C8CC70((__int64)&v45, v18, (__int64)v12, (__int64)v14, v15, v13);
        v13 = v49;
        if ( v16 == v17 )
          goto LABEL_10;
      }
      v19 = v46;
      v12 = &v46[HIDWORD(v47)];
      if ( v46 == v12 )
      {
LABEL_35:
        if ( HIDWORD(v47) >= (unsigned int)v47 )
          goto LABEL_33;
        ++v17;
        ++HIDWORD(v47);
        *v12 = v18;
        v13 = v49;
        ++v45;
        if ( v16 == v17 )
          goto LABEL_10;
      }
      else
      {
        while ( v18 != *v19 )
        {
          if ( v12 == ++v19 )
            goto LABEL_35;
        }
        if ( v16 == ++v17 )
        {
LABEL_10:
          v9 = v41;
          break;
        }
      }
    }
  }
  v51 = (__int64 *)v53;
  v52 = 0x800000000LL;
  v20 = *(__int64 **)a6;
  v44 = *(_QWORD *)a6 + 8LL * *(unsigned int *)(a6 + 8);
  if ( (__int64 *)v44 == v20 )
    goto LABEL_41;
  v42 = 0;
  v21 = v20;
  do
  {
    v22 = *v21;
    if ( !(unsigned __int8)sub_D0EBA0(v9, *v21, 0, a1, a3) )
      goto LABEL_22;
    v25 = (unsigned int)v52;
    v26 = (unsigned int)v52 + 1LL;
    if ( v26 > HIDWORD(v52) )
    {
      sub_C8D5F0((__int64)&v51, v53, v26, 8u, v23, v24);
      v25 = (unsigned int)v52;
    }
    v51[v25] = v22;
    v27 = *(_QWORD *)(v22 + 40);
    LODWORD(v52) = v52 + 1;
    if ( v49 )
    {
      v28 = v46;
      v29 = &v46[HIDWORD(v47)];
      if ( v46 != v29 )
      {
        while ( v27 != *v28 )
        {
          if ( v29 == ++v28 )
            goto LABEL_39;
        }
LABEL_21:
        ++v42;
        goto LABEL_22;
      }
    }
    else if ( sub_C8CA60((__int64)&v45, v27) )
    {
      goto LABEL_21;
    }
LABEL_39:
    if ( !(unsigned __int8)sub_D0EBA0(v9, v22, (__int64)&v45, a1, a3) )
      goto LABEL_21;
LABEL_22:
    ++v21;
  }
  while ( (__int64 *)v44 != v21 );
  v10 = a8;
  if ( v42 != (_DWORD)v52 )
  {
    v30 = v51;
    v31 = &v51[(unsigned int)v52];
    if ( v51 != v31 )
    {
      do
      {
        v32 = *v30++;
        a7(a8, v32);
      }
      while ( v31 != v30 );
      v31 = v51;
    }
    v33 = 0;
    goto LABEL_28;
  }
LABEL_41:
  v35 = *a5;
  v36 = (__int64)&(*a5)[*((unsigned int *)a5 + 2)];
  if ( *a5 != (__int64 *)v36 )
  {
    do
    {
      v37 = *v35++;
      a7(v10, v37);
    }
    while ( (__int64 *)v36 != v35 );
  }
  v31 = v51;
  v33 = 1;
LABEL_28:
  if ( v31 != (__int64 *)v53 )
    _libc_free((unsigned __int64)v31);
  if ( !v49 )
    _libc_free((unsigned __int64)v46);
  return v33;
}
