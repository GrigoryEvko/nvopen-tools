// Function: sub_3555D40
// Address: 0x3555d40
//
__int64 __fastcall sub_3555D40(__int64 a1, signed int a2, int a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // rdx
  unsigned int **v13; // rax
  unsigned int *v14; // r12
  __int64 v15; // r15
  int v16; // r14d
  char v17; // r13
  __int64 v18; // rsi
  unsigned int v19; // eax
  __int64 *v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  char v23; // al
  char v24; // r11
  int v25; // r13d
  unsigned int v26; // r8d
  __int64 v27; // r9
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // rsi
  int v31; // ecx
  unsigned int v32; // edx
  __int64 *v33; // rdi
  __int64 v34; // r10
  int **v36; // rax
  int *v37; // r12
  __int64 v38; // r14
  __int64 v39; // rsi
  __int64 v40; // rdi
  _QWORD *v41; // rax
  _BYTE *v42; // r12
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  unsigned __int64 v46; // rcx
  unsigned __int64 v47; // rsi
  int v48; // eax
  __int64 v49; // rcx
  __int64 v50; // r13
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rdx
  char *v54; // rdi
  int v55; // edi
  int v56; // r11d
  __int64 v57; // r15
  __int64 v58; // r13
  __int64 v59; // rdi
  __int64 v61; // [rsp+10h] [rbp-D0h]
  __int64 v62; // [rsp+18h] [rbp-C8h]
  char v63[8]; // [rsp+20h] [rbp-C0h]
  __int64 v68; // [rsp+48h] [rbp-98h] BYREF
  _BYTE v69[8]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v70; // [rsp+58h] [rbp-88h]
  unsigned int v71; // [rsp+68h] [rbp-78h]
  char *v72; // [rsp+70h] [rbp-70h]
  char v73; // [rsp+80h] [rbp-60h] BYREF

  v8 = a1 + 8;
  v68 = **(_QWORD **)(v8 - 8) + ((__int64)a2 << 8);
  sub_3554C70(v8, &v68);
  v11 = (unsigned int)a2;
  v12 = 1LL << a2;
  *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL * ((unsigned int)a2 >> 6)) |= 1LL << a2;
  v61 = 32LL * a2;
  v13 = (unsigned int **)(*(_QWORD *)(a1 + 784) + v61);
  v14 = *v13;
  v15 = (__int64)&(*v13)[*((unsigned int *)v13 + 2)];
  if ( *v13 == (unsigned int *)v15 )
    goto LABEL_27;
  *(_QWORD *)v63 = a2;
  v16 = a3;
  v62 = 4LL * a2;
  v17 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v18 = *v14;
      v19 = *(_DWORD *)(a1 + 1320);
      if ( v19 > unk_4CE009C )
      {
LABEL_13:
        v24 = v17;
        v25 = v16;
        if ( v24 )
          goto LABEL_14;
        v36 = (int **)(*(_QWORD *)(a1 + 784) + v61);
        v37 = *v36;
        v38 = (__int64)&(*v36)[*((unsigned int *)v36 + 2)];
        if ( (int *)v38 != *v36 )
        {
          while ( 1 )
          {
            while ( v25 > *v37 )
            {
LABEL_26:
              if ( (int *)v38 == ++v37 )
                goto LABEL_27;
            }
            v39 = v68;
            v40 = *(_QWORD *)(a1 + 128) + ((__int64)*v37 << 6);
            if ( *(_BYTE *)(v40 + 28) )
            {
              v41 = *(_QWORD **)(v40 + 8);
              v11 = *(unsigned int *)(v40 + 20);
              v12 = (__int64)&v41[v11];
              if ( v41 != (_QWORD *)v12 )
              {
                while ( v68 != *v41 )
                {
                  if ( (_QWORD *)v12 == ++v41 )
                    goto LABEL_30;
                }
                goto LABEL_26;
              }
LABEL_30:
              if ( (unsigned int)v11 >= *(_DWORD *)(v40 + 16) )
                goto LABEL_28;
              v11 = (unsigned int)(v11 + 1);
              ++v37;
              *(_DWORD *)(v40 + 20) = v11;
              *(_QWORD *)v12 = v39;
              ++*(_QWORD *)v40;
              if ( (int *)v38 == v37 )
                break;
            }
            else
            {
LABEL_28:
              sub_C8CC70(v40, v68, v12, v11, v9, v10);
              if ( (int *)v38 == ++v37 )
                break;
            }
          }
        }
LABEL_27:
        v26 = 0;
        goto LABEL_15;
      }
      if ( (int)v18 >= v16 )
        break;
LABEL_3:
      if ( (unsigned int *)v15 == ++v14 )
        goto LABEL_13;
    }
    if ( (_DWORD)v18 == v16 )
      break;
    v11 = (unsigned int)v18;
    v12 = (unsigned int)v18 >> 6;
    if ( (*(_QWORD *)(*(_QWORD *)(a1 + 56) + 8 * v12) & (1LL << v18)) != 0 )
      goto LABEL_3;
    v20 = *(__int64 **)(a1 + 1312);
    v21 = *v20;
    v22 = (v20[1] - *v20) >> 2;
    if ( (int)v18 >= v22 )
      sub_222CF80("vector::_M_range_check: __n (which is %zu) >= this->size() (which is %zu)", (int)v18);
    if ( *(_QWORD *)v63 >= v22 )
      sub_222CF80("vector::_M_range_check: __n (which is %zu) >= this->size() (which is %zu)", *(size_t *)v63, v22);
    v23 = sub_3555D40(
            a1,
            v18,
            (unsigned int)v16,
            a4,
            a5,
            (unsigned __int8)(a6 | (*(_DWORD *)(v21 + 4LL * (int)v18) < *(_DWORD *)(v21 + v62))));
    if ( v23 )
      v17 = v23;
    if ( (unsigned int *)v15 == ++v14 )
      goto LABEL_13;
  }
  if ( !a6 )
  {
    v42 = v69;
    sub_3555320(
      (__int64)v69,
      *(__int64 **)(a1 + 40),
      (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * *(unsigned int *)(a1 + 48)),
      a5);
    v45 = *(unsigned int *)(a4 + 8);
    v46 = *(unsigned int *)(a4 + 12);
    v47 = v45 + 1;
    v48 = *(_DWORD *)(a4 + 8);
    if ( v45 + 1 > v46 )
    {
      v57 = a4;
      v58 = *(_QWORD *)a4;
      v59 = a4;
      if ( *(_QWORD *)a4 > (unsigned __int64)v69 || (v57 = a4, v59 = a4, (unsigned __int64)v69 >= v58 + 88 * v45) )
      {
        sub_35498F0(v59, v47, v45, v46, v43, v44);
        v45 = *(unsigned int *)(v57 + 8);
        v49 = *(_QWORD *)v57;
        v48 = *(_DWORD *)(v57 + 8);
      }
      else
      {
        sub_35498F0(a4, v47, v45, v46, v43, v44);
        v49 = *(_QWORD *)a4;
        v45 = *(unsigned int *)(a4 + 8);
        v42 = &v69[*(_QWORD *)a4 - v58];
        v48 = *(_DWORD *)(a4 + 8);
      }
    }
    else
    {
      v49 = *(_QWORD *)a4;
    }
    v50 = v49 + 88 * v45;
    if ( v50 )
    {
      *(_QWORD *)(v50 + 16) = 0;
      *(_QWORD *)(v50 + 8) = 0;
      *(_DWORD *)(v50 + 24) = 0;
      *(_QWORD *)v50 = 1;
      v51 = *((_QWORD *)v42 + 1);
      ++*(_QWORD *)v42;
      v52 = *(_QWORD *)(v50 + 8);
      *(_QWORD *)(v50 + 8) = v51;
      LODWORD(v51) = *((_DWORD *)v42 + 4);
      *((_QWORD *)v42 + 1) = v52;
      LODWORD(v52) = *(_DWORD *)(v50 + 16);
      *(_DWORD *)(v50 + 16) = v51;
      LODWORD(v51) = *((_DWORD *)v42 + 5);
      *((_DWORD *)v42 + 4) = v52;
      LODWORD(v52) = *(_DWORD *)(v50 + 20);
      *(_DWORD *)(v50 + 20) = v51;
      v53 = *((unsigned int *)v42 + 6);
      *((_DWORD *)v42 + 5) = v52;
      LODWORD(v52) = *(_DWORD *)(v50 + 24);
      *(_DWORD *)(v50 + 24) = v53;
      *((_DWORD *)v42 + 6) = v52;
      *(_QWORD *)(v50 + 32) = v50 + 48;
      *(_QWORD *)(v50 + 40) = 0;
      if ( *((_DWORD *)v42 + 10) )
        sub_353DE10(v50 + 32, (char **)v42 + 4, v53, v49, v43, v44);
      *(_BYTE *)(v50 + 48) = v42[48];
      *(_QWORD *)(v50 + 52) = *(_QWORD *)(v42 + 52);
      *(_QWORD *)(v50 + 60) = *(_QWORD *)(v42 + 60);
      *(_QWORD *)(v50 + 72) = *((_QWORD *)v42 + 9);
      *(_DWORD *)(v50 + 80) = *((_DWORD *)v42 + 20);
      v48 = *(_DWORD *)(a4 + 8);
    }
    v54 = v72;
    *(_DWORD *)(a4 + 8) = v48 + 1;
    if ( v54 != &v73 )
      _libc_free((unsigned __int64)v54);
    sub_C7D6A0(v70, 8LL * v71, 8);
    v19 = *(_DWORD *)(a1 + 1320);
  }
  *(_DWORD *)(a1 + 1320) = v19 + 1;
LABEL_14:
  sub_35431F0(a1, a2);
  v26 = 1;
LABEL_15:
  v27 = *(_QWORD *)(a1 + 16);
  v28 = *(unsigned int *)(a1 + 48);
  v29 = *(_DWORD *)(a1 + 32);
  if ( v29 )
  {
    v30 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v28 - 8);
    v31 = v29 - 1;
    v32 = (v29 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
    v33 = (__int64 *)(v27 + 8LL * v32);
    v34 = *v33;
    if ( v30 == *v33 )
    {
LABEL_17:
      *v33 = -8192;
      LODWORD(v28) = *(_DWORD *)(a1 + 48);
      --*(_DWORD *)(a1 + 24);
      ++*(_DWORD *)(a1 + 28);
    }
    else
    {
      v55 = 1;
      while ( v34 != -4096 )
      {
        v56 = v55 + 1;
        v32 = v31 & (v55 + v32);
        v33 = (__int64 *)(v27 + 8LL * v32);
        v34 = *v33;
        if ( v30 == *v33 )
          goto LABEL_17;
        v55 = v56;
      }
    }
  }
  *(_DWORD *)(a1 + 48) = v28 - 1;
  return v26;
}
