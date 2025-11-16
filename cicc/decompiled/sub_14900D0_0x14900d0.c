// Function: sub_14900D0
// Address: 0x14900d0
//
void __fastcall sub_14900D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6)
{
  __int64 v6; // rdx
  __int64 v9; // r15
  __int64 *v10; // r14
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  char *v13; // r15
  __int64 v14; // rsi
  char *v15; // rsi
  char *v16; // r14
  __int64 v17; // rdx
  char *v18; // rdi
  __int64 v19; // r14
  __int64 v20; // rax
  char *v21; // r14
  unsigned __int64 v22; // rcx
  char *i; // r9
  __int64 v24; // r8
  char *j; // rax
  int v26; // edi
  int v27; // ecx
  __int64 v28; // rdx
  __int16 v29; // si
  _BYTE *v30; // rdi
  __int64 *v31; // r15
  __int64 v32; // rbx
  __int64 *v33; // r9
  __int64 v34; // rbx
  __int64 *v35; // r12
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // rbx
  __int16 v39; // ax
  __int64 v40; // rdx
  __int64 *v41; // rbx
  __int64 *k; // r10
  __int64 v43; // rax
  __int64 v44; // rdx
  _QWORD *v45; // rax
  char *v46; // rdi
  size_t v47; // rdx
  __int64 *v48; // [rsp+8h] [rbp-B8h]
  __int64 v49; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v50; // [rsp+28h] [rbp-98h]
  __int64 v51; // [rsp+28h] [rbp-98h]
  __int64 src; // [rsp+38h] [rbp-88h] BYREF
  __int64 *v54; // [rsp+40h] [rbp-80h] BYREF
  __int64 v55; // [rsp+48h] [rbp-78h]
  _BYTE v56[16]; // [rsp+50h] [rbp-70h] BYREF
  _BYTE *v57; // [rsp+60h] [rbp-60h] BYREF
  __int64 v58; // [rsp+68h] [rbp-58h]
  _BYTE v59[80]; // [rsp+70h] [rbp-50h] BYREF

  v6 = *(unsigned int *)(a2 + 8);
  src = a4;
  if ( (_DWORD)v6 && a4 )
  {
    v9 = *(_QWORD *)a2 + 8 * v6;
    v10 = *(__int64 **)a2;
    while ( 1 )
    {
      v11 = *v10;
      LOBYTE(v57) = 0;
      v58 = (__int64)sub_14525D0;
      sub_145E0E0(v11, (__int64)&v57);
      if ( (_BYTE)v57 )
        break;
      if ( (__int64 *)v9 == ++v10 )
        return;
    }
    v12 = *(unsigned int *)(a2 + 8);
    v13 = *(char **)a2;
    v14 = 8 * v12;
    if ( v12 > 1 )
    {
      qsort(*(void **)a2, v14 >> 3, 8u, (__compar_fn_t)sub_14525F0);
      v13 = *(char **)a2;
      v15 = (char *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
      if ( v15 != *(char **)a2 )
      {
LABEL_8:
        v16 = v13;
        do
        {
          v18 = v16;
          v16 += 8;
          if ( v15 == v16 )
            goto LABEL_11;
          v17 = *((_QWORD *)v16 - 1);
        }
        while ( v17 != *(_QWORD *)v16 );
        if ( v15 == v18 )
        {
          v16 = v15;
        }
        else
        {
          v45 = v18 + 16;
          if ( v15 != v18 + 16 )
          {
            while ( 1 )
            {
              if ( *v45 != v17 )
              {
                *((_QWORD *)v18 + 1) = *v45;
                v18 += 8;
              }
              if ( v15 == (char *)++v45 )
                break;
              v17 = *(_QWORD *)v18;
            }
            v13 = *(char **)a2;
            v46 = v18 + 8;
            v47 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - (_QWORD)v15;
            v16 = &v46[v47];
            if ( v15 != (char *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8)) )
            {
              memmove(v46, v15, v47);
              v13 = *(char **)a2;
            }
          }
        }
LABEL_11:
        v19 = (v16 - v13) >> 3;
        *(_DWORD *)(a2 + 8) = v19;
        v20 = 8LL * (unsigned int)v19;
        v21 = &v13[v20];
        if ( &v13[v20] != v13 )
        {
          v50 = v20;
          _BitScanReverse64(&v22, v20 >> 3);
          sub_1453740(v13, (__int64 *)&v13[v20], 2LL * (int)(63 - (v22 ^ 0x3F)));
          if ( v50 <= 0x80 )
          {
            sub_1453B30(v13, v21);
            goto LABEL_28;
          }
          sub_1453B30(v13, v13 + 128);
          for ( i = v13 + 128; v21 != i; i += 8 )
          {
            v24 = *(_QWORD *)i;
            for ( j = i; ; j -= 8 )
            {
              v28 = *((_QWORD *)j - 1);
              v29 = *(_WORD *)(v28 + 24);
              if ( *(_WORD *)(v24 + 24) == 5 )
              {
                v26 = *(_DWORD *)(v24 + 40);
                v27 = 1;
                if ( v29 != 5 )
                  goto LABEL_16;
                goto LABEL_21;
              }
              if ( v29 != 5 )
                break;
              v26 = 1;
LABEL_21:
              v27 = *(_DWORD *)(v28 + 40);
LABEL_16:
              if ( v27 >= v26 )
                break;
              *(_QWORD *)j = v28;
            }
            *(_QWORD *)j = v24;
          }
LABEL_28:
          if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) )
          {
            v51 = a2;
            v31 = *(__int64 **)a2;
            v32 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
            do
            {
              sub_148F0C0(a1, *v31, src, (__int64 *)&v54, (__int64 *)&v57, a5, a6);
              if ( !sub_14560B0((__int64)v54) )
                *v31 = (__int64)v54;
              ++v31;
            }
            while ( (__int64 *)v32 != v31 );
            v33 = *(__int64 **)v51;
            v34 = *(_QWORD *)v51 + 8LL * *(unsigned int *)(v51 + 8);
            v57 = v59;
            v58 = 0x400000000LL;
            if ( v33 != (__int64 *)v34 )
            {
              v35 = v33;
              v36 = v34;
              while ( 1 )
              {
                while ( 1 )
                {
                  v38 = *v35;
                  v39 = *(_WORD *)(*v35 + 24);
                  if ( v39 )
                    break;
LABEL_38:
                  if ( (__int64 *)v36 == ++v35 )
                    goto LABEL_52;
                }
                if ( v39 != 5 )
                  goto LABEL_35;
                v54 = (__int64 *)v56;
                v55 = 0x200000000LL;
                v40 = *(_QWORD *)(v38 + 40);
                v41 = *(__int64 **)(v38 + 32);
                for ( k = &v41[v40]; k != v41; LODWORD(v55) = v55 + 1 )
                {
                  while ( 1 )
                  {
                    v43 = *v41;
                    if ( *(_WORD *)(*v41 + 24) )
                      break;
                    if ( k == ++v41 )
                      goto LABEL_48;
                  }
                  v44 = (unsigned int)v55;
                  if ( (unsigned int)v55 >= HIDWORD(v55) )
                  {
                    v48 = k;
                    v49 = *v41;
                    sub_16CD150(&v54, v56, 0, 8);
                    v44 = (unsigned int)v55;
                    k = v48;
                    v43 = v49;
                  }
                  ++v41;
                  v54[v44] = v43;
                }
LABEL_48:
                v38 = sub_147EE30(a1, &v54, 0, 0, a5, a6);
                if ( v54 != (__int64 *)v56 )
                  _libc_free((unsigned __int64)v54);
                if ( v38 )
                {
LABEL_35:
                  v37 = (unsigned int)v58;
                  if ( (unsigned int)v58 >= HIDWORD(v58) )
                  {
                    sub_16CD150(&v57, v59, 0, 8);
                    v37 = (unsigned int)v58;
                  }
                  *(_QWORD *)&v57[8 * v37] = v38;
                  LODWORD(v58) = v58 + 1;
                  goto LABEL_38;
                }
                if ( (__int64 *)v36 == ++v35 )
                {
LABEL_52:
                  if ( (_DWORD)v58 && sub_148FD90(a1, (__int64)&v57, a3, a5, a6) )
                  {
                    sub_1458920(a3, &src);
                    v30 = v57;
                    if ( v57 == v59 )
                      return;
                  }
                  else
                  {
                    v30 = v57;
                    *(_DWORD *)(a3 + 8) = 0;
                    if ( v30 == v59 )
                      return;
                  }
                  _libc_free((unsigned __int64)v30);
                  return;
                }
              }
            }
          }
        }
LABEL_67:
        *(_DWORD *)(a3 + 8) = 0;
        return;
      }
    }
    else
    {
      v15 = &v13[v14];
      if ( v15 != v13 )
        goto LABEL_8;
    }
    *(_DWORD *)(a2 + 8) = 0;
    goto LABEL_67;
  }
}
