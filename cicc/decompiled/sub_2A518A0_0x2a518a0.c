// Function: sub_2A518A0
// Address: 0x2a518a0
//
void __fastcall sub_2A518A0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rax
  char v23; // al
  __int64 v24; // rax
  __int64 v25; // rax
  __m128i *v26; // rsi
  __int64 *v27; // r12
  __int64 *v28; // r15
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 **v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 **v36; // rax
  __int64 v37; // rsi
  __int64 v38; // r14
  __int64 v39; // rax
  __int64 *v40; // rbx
  __int64 v41; // r12
  __int64 v42; // r14
  unsigned __int64 v43; // rax
  _BYTE *v44; // rdx
  char v45; // si
  __int64 *v46; // r12
  __int64 *v47; // r13
  __int64 v48; // r15
  unsigned __int64 v49; // rax
  _BYTE *v50; // rcx
  __int64 v51; // [rsp+0h] [rbp-1F0h]
  __int64 v53; // [rsp+28h] [rbp-1C8h]
  __int64 v54; // [rsp+28h] [rbp-1C8h]
  __int64 v56; // [rsp+38h] [rbp-1B8h]
  __int64 v57; // [rsp+40h] [rbp-1B0h]
  __int64 v58; // [rsp+40h] [rbp-1B0h]
  __int64 v59; // [rsp+48h] [rbp-1A8h]
  __int64 v60; // [rsp+50h] [rbp-1A0h]
  __int64 v62; // [rsp+68h] [rbp-188h]
  __int64 v63; // [rsp+68h] [rbp-188h]
  char v64[32]; // [rsp+70h] [rbp-180h] BYREF
  __m128i v65; // [rsp+90h] [rbp-160h] BYREF
  char v66; // [rsp+A8h] [rbp-148h]
  __int64 v67; // [rsp+B0h] [rbp-140h]
  __int64 v68[4]; // [rsp+C0h] [rbp-130h] BYREF
  __int64 v69; // [rsp+E0h] [rbp-110h]
  __m128i v70; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v71; // [rsp+100h] [rbp-F0h] BYREF
  char v72; // [rsp+108h] [rbp-E8h]
  __int64 v73; // [rsp+110h] [rbp-E0h]
  _BYTE *v74; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+138h] [rbp-B8h]
  _BYTE v76[80]; // [rsp+140h] [rbp-B0h] BYREF
  char v77[8]; // [rsp+190h] [rbp-60h] BYREF
  int v78; // [rsp+198h] [rbp-58h] BYREF
  unsigned __int64 v79; // [rsp+1A0h] [rbp-50h]
  int *v80; // [rsp+1A8h] [rbp-48h]
  int *v81; // [rsp+1B0h] [rbp-40h]
  __int64 v82; // [rsp+1B8h] [rbp-38h]

  if ( !*(_DWORD *)(a1 + 8) && !*(_DWORD *)(a1 + 72) )
    return;
  v78 = 0;
  v74 = v76;
  v75 = 0x200000000LL;
  v80 = &v78;
  v81 = &v78;
  v79 = 0;
  v82 = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
    goto LABEL_32;
  v7 = sub_B91C10(a2, 38);
  if ( !v7 || (v8 = sub_AE94B0(v7), v53 = v9, v62 = v8, v8 == v9) )
  {
    v23 = *(_BYTE *)(a2 + 7) & 0x20;
    goto LABEL_14;
  }
  v51 = a5;
  v10 = v59;
  do
  {
    v11 = *(_QWORD *)(v62 + 24);
    sub_AF15E0((__int64)&v70, v11);
    sub_2A51690((__int64)v68, (__int64)&v74, &v70, v12, v13, v14);
    if ( !*(_BYTE *)(a4 + 28) )
    {
LABEL_57:
      sub_C8CC70(a4, v11, (__int64)v15, v16, v17, v18);
      goto LABEL_12;
    }
    v19 = *(__int64 **)(a4 + 8);
    v20 = *(unsigned int *)(a4 + 20);
    v15 = &v19[v20];
    if ( v19 == v15 )
    {
LABEL_58:
      if ( (unsigned int)v20 >= *(_DWORD *)(a4 + 16) )
        goto LABEL_57;
      *(_DWORD *)(a4 + 20) = v20 + 1;
      *v15 = v11;
      ++*(_QWORD *)a4;
    }
    else
    {
      while ( v11 != *v19 )
      {
        if ( v15 == ++v19 )
          goto LABEL_58;
      }
    }
LABEL_12:
    LOWORD(v10) = 0;
    v56 = sub_B10CD0(v11 + 48);
    v21 = *(_DWORD *)(v11 + 4) & 0x7FFFFFF;
    v60 = *(_QWORD *)(*(_QWORD *)(v11 + 32 * (2 - v21)) + 24LL);
    v57 = *(_QWORD *)(*(_QWORD *)(v11 + 32 * (1 - v21)) + 24LL);
    v22 = sub_B58EB0(v11, 0);
    sub_ADF2E0(a3, v22, v57, v60, v56, v60, v11 + 24, v10);
    v62 = *(_QWORD *)(v62 + 8);
  }
  while ( v62 != v53 );
  a5 = v51;
  v23 = *(_BYTE *)(a2 + 7) & 0x20;
LABEL_14:
  if ( v23 )
  {
    v24 = sub_B91C10(a2, 38);
    if ( v24 )
    {
      v25 = *(_QWORD *)(v24 + 8);
      v26 = (__m128i *)(v25 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v25 & 4) == 0 )
        v26 = 0;
      sub_B967C0(&v70, v26);
      v27 = (__int64 *)v70.m128i_i64[0];
      v54 = v70.m128i_i64[0] + 8LL * v70.m128i_u32[2];
      if ( v54 != v70.m128i_i64[0] )
      {
        while ( 1 )
        {
          v28 = (__int64 *)*v27;
          sub_AF48C0(v68, *v27);
          v66 = 0;
          v65.m128i_i64[0] = v68[0];
          v67 = v69;
          sub_2A51690((__int64)v64, (__int64)&v74, &v65, v29, v30, v31);
          if ( !*(_BYTE *)(a5 + 28) )
            goto LABEL_56;
          v36 = *(__int64 ***)(a5 + 8);
          v33 = *(unsigned int *)(a5 + 20);
          v32 = &v36[v33];
          if ( v36 != v32 )
          {
            while ( v28 != *v36 )
            {
              if ( v32 == ++v36 )
                goto LABEL_60;
            }
            goto LABEL_24;
          }
LABEL_60:
          if ( (unsigned int)v33 < *(_DWORD *)(a5 + 16) )
          {
            *(_DWORD *)(a5 + 20) = v33 + 1;
            *v32 = v28;
            ++*(_QWORD *)a5;
          }
          else
          {
LABEL_56:
            sub_C8CC70(a5, (__int64)v28, (__int64)v32, v33, v34, v35);
          }
LABEL_24:
          v37 = v28[3];
          v68[0] = v37;
          if ( v37 )
            sub_B96E90((__int64)v68, v37, 1);
          v58 = sub_B10CD0((__int64)v68);
          v63 = sub_B11F60((__int64)(v28 + 10));
          v38 = sub_B12000((__int64)(v28 + 9));
          v39 = sub_B12A50((__int64)v28, 0);
          sub_B14340(v39, v38, v63, v58, v28);
          if ( v68[0] )
            sub_B91220((__int64)v68, v68[0]);
          if ( (__int64 *)v54 == ++v27 )
          {
            v27 = (__int64 *)v70.m128i_i64[0];
            break;
          }
        }
      }
      if ( v27 != &v71 )
        _libc_free((unsigned __int64)v27);
    }
  }
LABEL_32:
  v40 = *(__int64 **)a1;
  v41 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v41 )
  {
    while ( 2 )
    {
      v42 = *v40;
      sub_AF15E0((__int64)&v70, *v40);
      if ( v82 )
      {
        if ( &v78 != (int *)sub_2A4E0F0((__int64)v77, (unsigned __int64 *)&v70) )
          goto LABEL_42;
      }
      else
      {
        v43 = (unsigned __int64)v74;
        v44 = &v74[40 * (unsigned int)v75];
        if ( v74 != v44 )
        {
          while ( 1 )
          {
            if ( *(_QWORD *)v43 == v70.m128i_i64[0] )
            {
              v45 = *(_BYTE *)(v43 + 24);
              if ( v45 == v72
                && (!v45 || *(_QWORD *)(v43 + 8) == v70.m128i_i64[1] && *(_QWORD *)(v43 + 16) == v71)
                && *(_QWORD *)(v43 + 32) == v73 )
              {
                break;
              }
            }
            v43 += 40LL;
            if ( v44 == (_BYTE *)v43 )
              goto LABEL_65;
          }
          if ( v44 != (_BYTE *)v43 )
          {
LABEL_42:
            if ( (__int64 *)v41 == ++v40 )
              goto LABEL_43;
            continue;
          }
        }
      }
      break;
    }
LABEL_65:
    sub_F519F0(v42, a2, a3);
    goto LABEL_42;
  }
LABEL_43:
  v46 = *(__int64 **)(a1 + 64);
  v47 = &v46[*(unsigned int *)(a1 + 72)];
  if ( v46 != v47 )
  {
    while ( 2 )
    {
      v48 = *v46;
      sub_AF48C0(v68, *v46);
      v72 = 0;
      v70.m128i_i64[0] = v68[0];
      v73 = v69;
      if ( v82 )
      {
        if ( &v78 != (int *)sub_2A4E0F0((__int64)v77, (unsigned __int64 *)&v70) )
          goto LABEL_52;
      }
      else
      {
        v49 = (unsigned __int64)v74;
        v50 = &v74[40 * (unsigned int)v75];
        if ( v74 != v50 )
        {
          while ( v68[0] != *(_QWORD *)v49 || *(_BYTE *)(v49 + 24) || v69 != *(_QWORD *)(v49 + 32) )
          {
            v49 += 40LL;
            if ( v50 == (_BYTE *)v49 )
              goto LABEL_63;
          }
          if ( v50 != (_BYTE *)v49 )
          {
LABEL_52:
            if ( v47 == ++v46 )
              goto LABEL_53;
            continue;
          }
        }
      }
      break;
    }
LABEL_63:
    sub_F51C80(v48, a2, a3);
    goto LABEL_52;
  }
LABEL_53:
  sub_2A4CE10(v79);
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
}
