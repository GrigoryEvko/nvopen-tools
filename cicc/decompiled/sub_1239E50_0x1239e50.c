// Function: sub_1239E50
// Address: 0x1239e50
//
__int64 __fastcall sub_1239E50(__int64 a1, __int64 *a2)
{
  __int64 v5; // rsi
  unsigned __int8 v6; // al
  __int64 v7; // r15
  __int64 v8; // r10
  unsigned int *v9; // r12
  unsigned int *v10; // r9
  unsigned int *v11; // r12
  __int64 v12; // rax
  unsigned int v13; // edi
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned int **v18; // r8
  __int64 v19; // r13
  __int64 v20; // r15
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // [rsp+0h] [rbp-B0h]
  unsigned __int8 v28; // [rsp+Fh] [rbp-A1h]
  __int64 v29; // [rsp+10h] [rbp-A0h]
  __int64 v30; // [rsp+10h] [rbp-A0h]
  __int64 v31; // [rsp+18h] [rbp-98h]
  unsigned int *v32; // [rsp+18h] [rbp-98h]
  unsigned int *v33; // [rsp+18h] [rbp-98h]
  _QWORD v34[2]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v35; // [rsp+30h] [rbp-80h]
  unsigned int *v36; // [rsp+40h] [rbp-70h] BYREF
  __int64 v37; // [rsp+48h] [rbp-68h] BYREF
  unsigned int v38; // [rsp+50h] [rbp-60h]
  __int64 v39; // [rsp+58h] [rbp-58h]
  unsigned int v40; // [rsp+60h] [rbp-50h]
  __int64 v41; // [rsp+68h] [rbp-48h]
  __int64 v42; // [rsp+70h] [rbp-40h]
  __int64 v43; // [rsp+78h] [rbp-38h]

  v31 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  v35 = 0;
  v34[0] = 0;
  v34[1] = 0;
  while ( 1 )
  {
    v36 = 0;
    sub_AADB10((__int64)&v37, 0x40u, 1);
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v28 = sub_12140B0(a1, (__int64 *)&v36, (__int64)v34);
    if ( v28 )
    {
      v23 = v42;
      v24 = v41;
      if ( v42 != v41 )
      {
        do
        {
          if ( *(_DWORD *)(v24 + 40) > 0x40u )
          {
            v25 = *(_QWORD *)(v24 + 32);
            if ( v25 )
              j_j___libc_free_0_0(v25);
          }
          if ( *(_DWORD *)(v24 + 24) > 0x40u )
          {
            v26 = *(_QWORD *)(v24 + 16);
            if ( v26 )
              j_j___libc_free_0_0(v26);
          }
          v24 += 48;
        }
        while ( v23 != v24 );
        v24 = v41;
      }
      if ( v24 )
        j_j___libc_free_0(v24, v43 - v24);
      if ( v40 > 0x40 && v39 )
        j_j___libc_free_0_0(v39);
      if ( v38 > 0x40 && v37 )
        j_j___libc_free_0_0(v37);
LABEL_49:
      v7 = v34[0];
      goto LABEL_50;
    }
    v5 = a2[1];
    if ( v5 == a2[2] )
    {
      sub_12142F0((__int64)a2, v5, (__int64)&v36);
      v19 = v42;
      v20 = v41;
    }
    else
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = v36;
        *(_DWORD *)(v5 + 16) = v38;
        *(_QWORD *)(v5 + 8) = v37;
        v38 = 0;
        *(_DWORD *)(v5 + 32) = v40;
        *(_QWORD *)(v5 + 24) = v39;
        v40 = 0;
        *(_QWORD *)(v5 + 40) = v41;
        *(_QWORD *)(v5 + 48) = v42;
        *(_QWORD *)(v5 + 56) = v43;
        a2[1] += 64;
        goto LABEL_10;
      }
      a2[1] = 64;
      v19 = v42;
      v20 = v41;
    }
    if ( v19 != v20 )
    {
      do
      {
        if ( *(_DWORD *)(v20 + 40) > 0x40u )
        {
          v21 = *(_QWORD *)(v20 + 32);
          if ( v21 )
            j_j___libc_free_0_0(v21);
        }
        if ( *(_DWORD *)(v20 + 24) > 0x40u )
        {
          v22 = *(_QWORD *)(v20 + 16);
          if ( v22 )
            j_j___libc_free_0_0(v22);
        }
        v20 += 48;
      }
      while ( v19 != v20 );
      v20 = v41;
    }
    if ( v20 )
      j_j___libc_free_0(v20, v43 - v20);
LABEL_10:
    if ( v40 > 0x40 && v39 )
      j_j___libc_free_0_0(v39);
    if ( v38 > 0x40 && v37 )
      j_j___libc_free_0_0(v37);
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
    *(_DWORD *)(a1 + 240) = sub_1205200(v31);
  }
  v6 = sub_120AFE0(a1, 13, "expected ')' here");
  v7 = v34[0];
  v28 = v6;
  if ( !v6 )
  {
    v8 = *a2;
    v27 = a2[1];
    if ( *a2 != v27 )
    {
      while ( 1 )
      {
        v9 = *(unsigned int **)(v8 + 40);
        v10 = *(unsigned int **)(v8 + 48);
        if ( v9 != v10 )
          break;
LABEL_48:
        v8 += 64;
        if ( v27 == v8 )
          goto LABEL_49;
      }
      v11 = v9 + 2;
      while ( 2 )
      {
        if ( (*(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL) == 0xFFFFFFFFFFFFFFF8LL )
        {
          v12 = *(_QWORD *)(a1 + 1544);
          if ( !v12 )
          {
            v14 = a1 + 1536;
            goto LABEL_31;
          }
          v13 = *(_DWORD *)v7;
          v14 = a1 + 1536;
          do
          {
            while ( 1 )
            {
              v15 = *(_QWORD *)(v12 + 16);
              v16 = *(_QWORD *)(v12 + 24);
              if ( *(_DWORD *)(v12 + 32) >= v13 )
                break;
              v12 = *(_QWORD *)(v12 + 24);
              if ( !v16 )
                goto LABEL_29;
            }
            v14 = v12;
            v12 = *(_QWORD *)(v12 + 16);
          }
          while ( v15 );
LABEL_29:
          if ( v14 == a1 + 1536 || v13 < *(_DWORD *)(v14 + 32) )
          {
LABEL_31:
            v36 = (unsigned int *)v7;
            v29 = v8;
            v32 = v10;
            v17 = sub_12395C0((_QWORD *)(a1 + 1528), v14, &v36);
            v8 = v29;
            v10 = v32;
            v14 = v17;
          }
          v36 = v11;
          v18 = *(unsigned int ***)(v14 + 48);
          if ( v18 == *(unsigned int ***)(v14 + 56) )
          {
            v30 = v8;
            v33 = v10;
            sub_1214860((const __m128i **)(v14 + 40), *(const __m128i **)(v14 + 48), &v36, (_QWORD *)(v7 + 8));
            v8 = v30;
            v10 = v33;
          }
          else
          {
            if ( v18 )
            {
              *v18 = v11;
              v18[1] = *(unsigned int **)(v7 + 8);
              v18 = *(unsigned int ***)(v14 + 48);
            }
            *(_QWORD *)(v14 + 48) = v18 + 2;
          }
        }
        v7 += 16;
        if ( v10 == v11 + 10 )
          goto LABEL_48;
        v11 += 12;
        continue;
      }
    }
  }
LABEL_50:
  if ( v7 )
    j_j___libc_free_0(v7, v35 - v7);
  return v28;
}
