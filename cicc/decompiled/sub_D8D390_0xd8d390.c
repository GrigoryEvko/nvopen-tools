// Function: sub_D8D390
// Address: 0xd8d390
//
__int64 __fastcall sub_D8D390(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v6; // r9
  __int64 v7; // rdx
  unsigned __int64 v8; // rdx
  __int64 *v9; // rsi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  __int64 *v13; // r14
  unsigned __int64 v14; // r13
  int v15; // r9d
  int v16; // r12d
  __int64 v17; // rdx
  __int64 v18; // r11
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // r10
  __int64 v23; // rdi
  bool v24; // cc
  __int64 v25; // rdi
  __int64 v26; // rdi
  unsigned __int64 i; // rbx
  unsigned __int64 v28; // rcx
  __int64 v29; // r13
  __int64 v30; // r14
  unsigned __int64 v31; // rbx
  unsigned int v32; // edi
  unsigned int v33; // ecx
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r10
  __int64 *v38; // rbx
  unsigned int v39; // edx
  unsigned int v40; // esi
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rdi
  __int64 v47; // r13
  __int64 *v48; // [rsp+8h] [rbp-C8h]
  __int64 v49; // [rsp+10h] [rbp-C0h]
  __int64 v50; // [rsp+20h] [rbp-B0h]
  int v51; // [rsp+28h] [rbp-A8h]
  __int64 v52; // [rsp+28h] [rbp-A8h]
  __int64 v53; // [rsp+28h] [rbp-A8h]
  __int64 v54; // [rsp+30h] [rbp-A0h]
  int v55; // [rsp+30h] [rbp-A0h]
  unsigned int v56; // [rsp+30h] [rbp-A0h]
  __int64 *v57; // [rsp+38h] [rbp-98h]
  __int64 *v58; // [rsp+38h] [rbp-98h]
  __int64 *v59; // [rsp+38h] [rbp-98h]
  __int64 v60; // [rsp+38h] [rbp-98h]
  __int64 v61; // [rsp+48h] [rbp-88h]
  __int64 v62; // [rsp+70h] [rbp-60h] BYREF
  __int64 v63; // [rsp+78h] [rbp-58h]
  __int64 v64; // [rsp+80h] [rbp-50h]
  unsigned int v65; // [rsp+88h] [rbp-48h]
  __int64 v66; // [rsp+90h] [rbp-40h]
  unsigned int v67; // [rsp+98h] [rbp-38h]

  result = (__int64)a2 - a1;
  v49 = a3;
  v57 = a2;
  if ( (__int64)a2 - a1 <= 768 )
    return result;
  v6 = a2;
  v7 = (__int64)a2 - a1;
  if ( !a3 )
    goto LABEL_46;
  v48 = (__int64 *)(a1 + 48);
  while ( 2 )
  {
    --v49;
    v8 = *(_QWORD *)(a1 + 48);
    v9 = (__int64 *)(a1
                   + 16
                   * (((0xAAAAAAAAAAAAAAABLL * (((__int64)v57 - a1) >> 4)
                      + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v57 - a1) >> 4)) >> 63))
                     & 0xFFFFFFFFFFFFFFFELL)
                    + (__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)v57 - a1) >> 4)) / 2));
    v10 = *v9;
    if ( v8 >= *v9
      && (v8 != *v9
       || *(_QWORD *)(*(_QWORD *)(a1 + 56) & 0xFFFFFFFFFFFFFFF8LL) >= *(_QWORD *)(v9[1] & 0xFFFFFFFFFFFFFFF8LL)) )
    {
      v28 = *(v57 - 6);
      if ( v8 >= v28
        && (v8 != v28
         || *(_QWORD *)(*(_QWORD *)(a1 + 56) & 0xFFFFFFFFFFFFFFF8LL) >= *(_QWORD *)(*(v57 - 5) & 0xFFFFFFFFFFFFFFF8LL)) )
      {
        if ( v10 >= v28 )
        {
          if ( v10 == v28 && *(_QWORD *)(v9[1] & 0xFFFFFFFFFFFFFFF8LL) < *(_QWORD *)(*(v57 - 5) & 0xFFFFFFFFFFFFFFF8LL) )
            v9 = v57 - 6;
          goto LABEL_13;
        }
        goto LABEL_12;
      }
LABEL_41:
      sub_D8D280((__int64 *)a1, v48);
      goto LABEL_14;
    }
    v11 = *(v57 - 6);
    if ( v10 < v11
      || v10 == v11 && *(_QWORD *)(v9[1] & 0xFFFFFFFFFFFFFFF8LL) < *(_QWORD *)(*(v57 - 5) & 0xFFFFFFFFFFFFFFF8LL) )
    {
      goto LABEL_13;
    }
    if ( v8 >= v11
      && (v8 != v11
       || *(_QWORD *)(*(_QWORD *)(a1 + 56) & 0xFFFFFFFFFFFFFFF8LL) >= *(_QWORD *)(*(v57 - 5) & 0xFFFFFFFFFFFFFFF8LL)) )
    {
      goto LABEL_41;
    }
LABEL_12:
    v9 = v57 - 6;
LABEL_13:
    sub_D8D280((__int64 *)a1, v9);
LABEL_14:
    v12 = *(_QWORD *)a1;
    v13 = v48;
    v14 = (unsigned __int64)v57;
    while ( 1 )
    {
      v22 = *v13;
      if ( *v13 < v12
        || *v13 == v12
        && *(_QWORD *)(v13[1] & 0xFFFFFFFFFFFFFFF8LL) < *(_QWORD *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL) )
      {
        goto LABEL_27;
      }
      for ( i = v14 - 48; ; i -= 48LL )
      {
        v14 = i;
        if ( *(_QWORD *)i <= v12
          && (*(_QWORD *)i != v12
           || *(_QWORD *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL) >= *(_QWORD *)(*(_QWORD *)(i + 8)
                                                                                   & 0xFFFFFFFFFFFFFFF8LL)) )
        {
          break;
        }
      }
      if ( (unsigned __int64)v13 >= i )
        break;
      v15 = *((_DWORD *)v13 + 6);
      v16 = *((_DWORD *)v13 + 10);
      *((_DWORD *)v13 + 6) = 0;
      v17 = v13[1];
      v18 = v13[2];
      *((_DWORD *)v13 + 10) = 0;
      v19 = *(_QWORD *)i;
      v20 = v13[4];
      v63 = v17;
      *v13 = v19;
      v13[1] = *(_QWORD *)(i + 8);
      v13[2] = *(_QWORD *)(i + 16);
      *((_DWORD *)v13 + 6) = *(_DWORD *)(i + 24);
      *(_DWORD *)(i + 24) = 0;
      if ( *((_DWORD *)v13 + 10) > 0x40u )
      {
        v21 = v13[4];
        if ( v21 )
        {
          v50 = v18;
          v51 = v15;
          v54 = v22;
          j_j___libc_free_0_0(v21);
          v18 = v50;
          v15 = v51;
          v22 = v54;
        }
      }
      v13[4] = *(_QWORD *)(i + 32);
      *((_DWORD *)v13 + 10) = *(_DWORD *)(i + 40);
      v23 = v63;
      v24 = *(_DWORD *)(i + 24) <= 0x40u;
      *(_DWORD *)(i + 40) = 0;
      *(_QWORD *)i = v22;
      *(_QWORD *)(i + 8) = v23;
      if ( v24 || (v25 = *(_QWORD *)(i + 16)) == 0 )
      {
        *(_QWORD *)(i + 16) = v18;
        *(_DWORD *)(i + 24) = v15;
      }
      else
      {
        v52 = v18;
        v55 = v15;
        j_j___libc_free_0_0(v25);
        v24 = *(_DWORD *)(i + 40) <= 0x40u;
        *(_QWORD *)(i + 16) = v52;
        *(_DWORD *)(i + 24) = v55;
        if ( !v24 )
        {
          v26 = *(_QWORD *)(i + 32);
          if ( v26 )
            j_j___libc_free_0_0(v26);
        }
      }
      *(_QWORD *)(i + 32) = v20;
      *(_DWORD *)(i + 40) = v16;
      v12 = *(_QWORD *)a1;
LABEL_27:
      v13 += 6;
    }
    result = sub_D8D390(v13, v57, v49);
    v6 = v13;
    v7 = (__int64)v13 - a1;
    if ( (__int64)v13 - a1 > 768 )
    {
      if ( v49 )
      {
        v57 = v13;
        continue;
      }
LABEL_46:
      v29 = 0xAAAAAAAAAAAAAAABLL * (v7 >> 4);
      v30 = (v29 - 2) >> 1;
      v31 = a1 + 16 * (v30 + ((v29 - 2) & 0xFFFFFFFFFFFFFFFELL));
      while ( 1 )
      {
        v32 = *(_DWORD *)(v31 + 24);
        v33 = *(_DWORD *)(v31 + 40);
        *(_DWORD *)(v31 + 24) = 0;
        v34 = *(_QWORD *)(v31 + 16);
        v35 = *(_QWORD *)(v31 + 32);
        *(_DWORD *)(v31 + 40) = 0;
        v36 = *(_QWORD *)(v31 + 8);
        v37 = *(_QWORD *)v31;
        v65 = v32;
        v64 = v34;
        v67 = v33;
        v66 = v35;
        v58 = v6;
        v62 = v37;
        v63 = v36;
        sub_D87410(a1, v30, v29, &v62);
        v6 = v58;
        if ( v67 > 0x40 && v66 )
        {
          j_j___libc_free_0_0(v66);
          v6 = v58;
        }
        if ( v65 > 0x40 && v64 )
        {
          v59 = v6;
          j_j___libc_free_0_0(v64);
          v6 = v59;
        }
        v31 -= 48LL;
        if ( !v30 )
          break;
        --v30;
      }
      v38 = v6 - 6;
      do
      {
        v39 = *((_DWORD *)v38 + 6);
        v40 = *((_DWORD *)v38 + 10);
        *((_DWORD *)v38 + 6) = 0;
        v41 = *v38;
        v42 = v38[1];
        *((_DWORD *)v38 + 10) = 0;
        v43 = v38[4];
        v61 = v42;
        v44 = v38[2];
        *v38 = *(_QWORD *)a1;
        v38[1] = *(_QWORD *)(a1 + 8);
        v38[2] = *(_QWORD *)(a1 + 16);
        *((_DWORD *)v38 + 6) = *(_DWORD *)(a1 + 24);
        *(_DWORD *)(a1 + 24) = 0;
        if ( *((_DWORD *)v38 + 10) > 0x40u )
        {
          v45 = v38[4];
          if ( v45 )
          {
            v53 = v44;
            v56 = v39;
            v60 = v41;
            j_j___libc_free_0_0(v45);
            v44 = v53;
            v39 = v56;
            v41 = v60;
          }
        }
        v46 = *(_QWORD *)(a1 + 32);
        v66 = v43;
        v47 = (__int64)v38 - a1;
        v65 = v39;
        v38[4] = v46;
        LODWORD(v46) = *(_DWORD *)(a1 + 40);
        v62 = v41;
        *((_DWORD *)v38 + 10) = v46;
        *(_DWORD *)(a1 + 40) = 0;
        v63 = v61;
        v67 = v40;
        v64 = v44;
        result = sub_D87410(a1, 0, 0xAAAAAAAAAAAAAAABLL * (((__int64)v38 - a1) >> 4), &v62);
        if ( v67 > 0x40 && v66 )
          result = j_j___libc_free_0_0(v66);
        if ( v65 > 0x40 && v64 )
          result = j_j___libc_free_0_0(v64);
        v38 -= 6;
      }
      while ( v47 > 48 );
    }
    return result;
  }
}
