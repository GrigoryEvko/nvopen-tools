// Function: sub_1241420
// Address: 0x1241420
//
__int64 __fastcall sub_1241420(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  char v7; // bl
  char **v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // r14
  unsigned __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rbx
  unsigned __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rbx
  char v21; // al
  __int64 v22; // rdi
  __int64 v23; // r15
  __int64 v24; // r13
  __int64 v25; // r15
  __int64 v26; // rdi
  __int64 v27; // rbx
  __int64 v28; // r13
  __int64 v29; // rdi
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // rdi
  __int64 *v34; // r15
  __int64 *v35; // r13
  __int64 v36; // rdi
  __int64 v37; // [rsp+18h] [rbp-168h]
  unsigned __int64 v38; // [rsp+28h] [rbp-158h]
  __int64 v40; // [rsp+40h] [rbp-140h] BYREF
  __int64 v41; // [rsp+48h] [rbp-138h]
  __int64 v42; // [rsp+50h] [rbp-130h]
  _BYTE *v43; // [rsp+60h] [rbp-120h] BYREF
  __int64 v44; // [rsp+68h] [rbp-118h]
  unsigned __int64 v45; // [rsp+70h] [rbp-110h]
  _BYTE v46[40]; // [rsp+78h] [rbp-108h] BYREF
  char *v47; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v48; // [rsp+A8h] [rbp-D8h]
  __int64 v49; // [rsp+B0h] [rbp-D0h]
  _BYTE v50[40]; // [rsp+B8h] [rbp-C8h] BYREF
  char *v51; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v52; // [rsp+E8h] [rbp-98h]
  __int64 v53; // [rsp+F0h] [rbp-90h]
  _BYTE v54[40]; // [rsp+F8h] [rbp-88h] BYREF
  __int64 v55; // [rsp+120h] [rbp-60h]
  __int64 v56; // [rsp+128h] [rbp-58h]
  unsigned __int64 v57; // [rsp+130h] [rbp-50h]
  __int64 *v58; // [rsp+138h] [rbp-48h]
  __int64 *v59; // [rsp+140h] [rbp-40h]
  __int64 v60; // [rsp+148h] [rbp-38h]

  while ( !(unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in alloc")
       && !(unsigned __int8)sub_120AFE0(a1, 496, "expected 'versions' in alloc")
       && !(unsigned __int8)sub_120AFE0(a1, 16, "expected ':'")
       && !(unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in versions") )
  {
    v44 = 0;
    v45 = 40;
    v43 = v46;
    while ( 1 )
    {
      v3 = (__int64)&v51;
      LOBYTE(v51) = 0;
      if ( (unsigned __int8)sub_12123F0(a1, &v51) )
        goto LABEL_66;
      v6 = v44;
      v7 = (char)v51;
      if ( v44 + 1 > v45 )
      {
        sub_C8D290((__int64)&v43, v46, v44 + 1, 1u, v4, v5);
        v6 = v44;
      }
      v43[v6] = v7;
      ++v44;
      if ( *(_DWORD *)(a1 + 240) != 4 )
        break;
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    }
    v3 = 13;
    if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' in versions") )
      goto LABEL_66;
    v3 = 4;
    if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' in alloc") )
      goto LABEL_66;
    v3 = (__int64)&v40;
    v8 = (char **)a1;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    if ( (unsigned __int8)sub_1240FB0(a1, &v40) )
      goto LABEL_59;
    v48 = 0;
    v47 = v50;
    v49 = 40;
    if ( v44 )
    {
      v3 = (__int64)&v43;
      v8 = &v47;
      sub_1205920((__int64)&v47, (__int64)&v43, v9, v10, v11, v12);
    }
    v13 = v41;
    v14 = v40;
    v15 = v41 - v40;
    if ( v41 == v40 )
    {
      v17 = 0;
    }
    else
    {
      if ( v15 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(v8, v3, v9);
      v16 = sub_22077B0(v41 - v40);
      v13 = v41;
      v14 = v40;
      v17 = v16;
    }
    v18 = v17 + v15;
    v19 = v17;
    v38 = v18;
    if ( v13 != v14 )
    {
      v37 = v17;
      v20 = v13;
      do
      {
        while ( 1 )
        {
          if ( v19 )
          {
            v21 = *(_BYTE *)v14;
            *(_DWORD *)(v19 + 16) = 0;
            *(_DWORD *)(v19 + 20) = 12;
            *(_BYTE *)v19 = v21;
            *(_QWORD *)(v19 + 8) = v19 + 24;
            if ( *(_DWORD *)(v14 + 16) )
              break;
          }
          v14 += 72;
          v19 += 72;
          if ( v20 == v14 )
            goto LABEL_24;
        }
        v3 = v14 + 8;
        v22 = v19 + 8;
        v14 += 72;
        v19 += 72;
        sub_1205840(v22, v3, v9, v13, v11, v12);
      }
      while ( v20 != v14 );
LABEL_24:
      v17 = v37;
    }
    v52 = 0;
    v51 = v54;
    v53 = 40;
    if ( v48 )
    {
      v3 = (__int64)&v47;
      sub_1205CB0((__int64)&v51, &v47, v9, v13, v11, v12);
    }
    v56 = v19;
    v55 = v17;
    v57 = v38;
    v58 = 0;
    v59 = 0;
    v23 = *(_QWORD *)(a2 + 8);
    v60 = 0;
    if ( v23 == *(_QWORD *)(a2 + 16) )
    {
      v3 = v23;
      sub_9EB9D0(a2, (char *)v23, (__int64)&v51);
      v34 = v59;
      v35 = v58;
      if ( v58 != v59 )
      {
        do
        {
          v36 = *v35;
          if ( *v35 )
          {
            v3 = v35[2] - v36;
            j_j___libc_free_0(v36, v3);
          }
          v35 += 3;
        }
        while ( v35 != v34 );
        v34 = v58;
      }
      if ( v34 )
      {
        v3 = v60 - (_QWORD)v34;
        j_j___libc_free_0(v34, v60 - (_QWORD)v34);
      }
    }
    else
    {
      if ( v23 )
      {
        *(_QWORD *)(v23 + 8) = 0;
        *(_QWORD *)v23 = v23 + 24;
        *(_QWORD *)(v23 + 16) = 40;
        if ( v52 )
        {
          v3 = (__int64)&v51;
          sub_1205CB0(v23, &v51, v9, v13, v11, v12);
        }
        *(_QWORD *)(v23 + 64) = v55;
        *(_QWORD *)(v23 + 72) = v56;
        *(_QWORD *)(v23 + 80) = v57;
        v57 = 0;
        v56 = 0;
        v55 = 0;
        *(_QWORD *)(v23 + 88) = v58;
        *(_QWORD *)(v23 + 96) = v59;
        *(_QWORD *)(v23 + 104) = v60;
        v23 = *(_QWORD *)(a2 + 8);
      }
      *(_QWORD *)(a2 + 8) = v23 + 112;
    }
    v24 = v56;
    v25 = v55;
    if ( v56 != v55 )
    {
      do
      {
        v26 = *(_QWORD *)(v25 + 8);
        if ( v26 != v25 + 24 )
          _libc_free(v26, v3);
        v25 += 72;
      }
      while ( v24 != v25 );
      v25 = v55;
    }
    if ( v25 )
    {
      v3 = v57 - v25;
      j_j___libc_free_0(v25, v57 - v25);
    }
    if ( v51 != v54 )
      _libc_free(v51, v3);
    if ( v47 != v50 )
      _libc_free(v47, v3);
    v3 = 13;
    if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' in alloc") )
    {
LABEL_59:
      v31 = v41;
      v32 = v40;
      if ( v41 != v40 )
      {
        do
        {
          v33 = *(_QWORD *)(v32 + 8);
          if ( v33 != v32 + 24 )
            _libc_free(v33, v3);
          v32 += 72;
        }
        while ( v31 != v32 );
        v32 = v40;
      }
      if ( v32 )
      {
        v3 = v42 - v32;
        j_j___libc_free_0(v32, v42 - v32);
      }
LABEL_66:
      if ( v43 != v46 )
        _libc_free(v43, v3);
      return 1;
    }
    v27 = v41;
    v28 = v40;
    if ( v41 != v40 )
    {
      do
      {
        v29 = *(_QWORD *)(v28 + 8);
        if ( v29 != v28 + 24 )
          _libc_free(v29, 13);
        v28 += 72;
      }
      while ( v27 != v28 );
      v28 = v40;
    }
    if ( v28 )
    {
      v3 = v42 - v28;
      j_j___libc_free_0(v28, v42 - v28);
    }
    if ( v43 != v46 )
      _libc_free(v43, v3);
    if ( *(_DWORD *)(a1 + 240) != 4 )
      return sub_120AFE0(a1, 13, "expected ')' in allocs");
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  return 1;
}
