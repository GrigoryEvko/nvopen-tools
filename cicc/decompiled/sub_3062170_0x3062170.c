// Function: sub_3062170
// Address: 0x3062170
//
__int64 __fastcall sub_3062170(int *a1, __int64 *a2, unsigned __int64 *a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v7; // rdi
  __int16 v8; // bx
  __int64 v9; // rax
  char *v10; // rsi
  int v11; // eax
  __int64 v12; // r9
  __int64 v13; // r15
  unsigned __int64 v14; // rax
  __int64 i; // rbx
  __int64 v16; // rax
  size_t v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r14
  __int64 *v20; // r13
  _QWORD *v21; // rax
  _QWORD *v22; // r15
  int v23; // ecx
  __int64 v24; // r10
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 j; // rbx
  __int64 v30; // rax
  size_t v31; // rdx
  __int64 v32; // r8
  __int64 v33; // rcx
  size_t *v34; // r8
  size_t **v35; // r14
  char *v36; // rsi
  unsigned __int64 v37; // r8
  __int64 v38; // r12
  __int64 v39; // rbx
  _QWORD *v40; // rdi
  unsigned __int64 v41; // r8
  __int64 v42; // r12
  __int64 v43; // rbx
  _QWORD *v44; // rdi
  __int64 v45; // [rsp+0h] [rbp-B0h]
  __int64 v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v48; // [rsp+10h] [rbp-A0h]
  __int64 v49; // [rsp+10h] [rbp-A0h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  __int64 v51; // [rsp+18h] [rbp-98h]
  __int64 v52; // [rsp+20h] [rbp-90h]
  __int64 v53; // [rsp+20h] [rbp-90h]
  size_t n; // [rsp+28h] [rbp-88h]
  size_t na; // [rsp+28h] [rbp-88h]
  _QWORD *v56; // [rsp+30h] [rbp-80h] BYREF
  int v57; // [rsp+38h] [rbp-78h] BYREF
  char v58; // [rsp+3Ch] [rbp-74h]
  unsigned __int64 v59; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v60; // [rsp+48h] [rbp-68h]
  __int64 v61; // [rsp+4Ch] [rbp-64h]
  __int64 v62; // [rsp+60h] [rbp-50h] BYREF
  __int128 v63; // [rsp+68h] [rbp-48h]

  v4 = a2[1];
  v5 = *a2;
  if ( v4 == 15 )
  {
    if ( *(_QWORD *)v5 == 0x2D636972656E6567LL
      && *(_DWORD *)(v5 + 8) == 1848471412
      && *(_WORD *)(v5 + 12) == 30326
      && *(_BYTE *)(v5 + 14) == 109 )
    {
      v7 = sub_22077B0(0x10u);
      if ( v7 )
        *(_QWORD *)v7 = &unk_4A30EE0;
      goto LABEL_15;
    }
    return 0;
  }
  if ( v4 != 21 )
  {
    if ( v4 == 32 )
    {
      if ( *(_QWORD *)v5 ^ 0x65732D787470766ELL | *(_QWORD *)(v5 + 8) ^ 0x6C61626F6C672D74LL
        || *(_QWORD *)(v5 + 16) ^ 0x612D79617272612DLL | *(_QWORD *)(v5 + 24) ^ 0x746E656D6E67696CLL )
      {
        return 0;
      }
      sub_36CD370(&v59);
      v8 = v59;
      v9 = sub_22077B0(0x10u);
      v7 = v9;
      if ( v9 )
      {
        *(_WORD *)(v9 + 8) = v8;
        *(_QWORD *)v9 = &unk_4A30F60;
      }
      goto LABEL_15;
    }
    if ( v4 != 12 || *(_QWORD *)v5 != 0x6665722D6D76766ELL || *(_DWORD *)(v5 + 8) != 1952671084 )
      return 0;
    v11 = *a1;
    v58 = 0;
    v57 = v11;
    sub_305F6F0((__int64)&v59, &v57);
    v62 = 0;
    *(_QWORD *)&v63 = 0;
    *((_QWORD *)&v63 + 1) = 0x1000000000LL;
    if ( (_DWORD)v61 )
    {
      sub_C92620((__int64)&v62, v60);
      v12 = v62;
      v13 = 8LL * (unsigned int)v63 + 8;
      v50 = v62;
      v48 = v59;
      *(_QWORD *)((char *)&v63 + 4) = v61;
      if ( (_DWORD)v63 )
      {
        v52 = 8LL * (unsigned int)(v63 - 1);
        v14 = v59;
        for ( i = 0; ; i += 8 )
        {
          v19 = *(_QWORD *)(v14 + i);
          v20 = (__int64 *)(v12 + i);
          if ( v19 == -8 || !v19 )
          {
            *v20 = v19;
          }
          else
          {
            n = *(_QWORD *)v19;
            v16 = sub_C7D670(*(_QWORD *)v19 + 17LL, 8);
            v17 = n;
            v18 = v16;
            if ( n )
            {
              v47 = v16;
              memcpy((void *)(v16 + 16), (const void *)(v19 + 16), n);
              v17 = n;
              v18 = v47;
            }
            *(_BYTE *)(v18 + v17 + 16) = 0;
            *(_QWORD *)v18 = v17;
            *(_DWORD *)(v18 + 8) = *(_DWORD *)(v19 + 8);
            *v20 = v18;
            *(_DWORD *)(v50 + v13) = *(_DWORD *)(v48 + v13);
          }
          v13 += 4;
          if ( v52 == i )
            break;
          v14 = v59;
          v12 = v62;
        }
      }
    }
    v21 = (_QWORD *)sub_22077B0(0x20u);
    v22 = v21;
    if ( v21 )
    {
      v23 = DWORD1(v63);
      v21[1] = 0;
      v21[2] = 0;
      *v21 = &unk_4A30FA0;
      v21[3] = 0x1000000000LL;
      if ( v23 )
      {
        sub_C92620((__int64)(v21 + 1), v63);
        v24 = v22[1];
        v25 = v62;
        v26 = *((unsigned int *)v22 + 4);
        v27 = 8 * v26 + 8;
        v49 = v24;
        v46 = v62;
        *(_QWORD *)((char *)v22 + 20) = *(_QWORD *)((char *)&v63 + 4);
        if ( (_DWORD)v26 )
        {
          v51 = 8LL * (unsigned int)(v26 - 1);
          v28 = v25;
          for ( j = 0; ; j += 8 )
          {
            v34 = *(size_t **)(v28 + j);
            v35 = (size_t **)(v24 + j);
            if ( v34 == (size_t *)-8LL || !v34 )
            {
              *v35 = v34;
            }
            else
            {
              v53 = *(_QWORD *)(v28 + j);
              na = *v34;
              v30 = sub_C7D670(*v34 + 17, 8);
              v31 = na;
              v32 = v53;
              v33 = v30;
              if ( na )
              {
                v45 = v30;
                memcpy((void *)(v30 + 16), (const void *)(v53 + 16), na);
                v31 = na;
                v32 = v53;
                v33 = v45;
              }
              *(_BYTE *)(v33 + v31 + 16) = 0;
              *(_QWORD *)v33 = v31;
              *(_DWORD *)(v33 + 8) = *(_DWORD *)(v32 + 8);
              *v35 = (size_t *)v33;
              *(_DWORD *)(v49 + v27) = *(_DWORD *)(v46 + v27);
            }
            v27 += 4;
            if ( v51 == j )
              break;
            v28 = v62;
            v24 = v22[1];
          }
        }
      }
    }
    v56 = v22;
    v36 = (char *)a3[1];
    if ( v36 == (char *)a3[2] )
    {
      sub_2275C60(a3, v36, &v56);
      v22 = v56;
    }
    else
    {
      if ( v36 )
      {
        *(_QWORD *)v36 = v22;
        a3[1] += 8LL;
LABEL_57:
        v37 = v62;
        if ( DWORD1(v63) && (_DWORD)v63 )
        {
          v38 = 8LL * (unsigned int)v63;
          v39 = 0;
          do
          {
            v40 = *(_QWORD **)(v37 + v39);
            if ( v40 != (_QWORD *)-8LL && v40 )
            {
              sub_C7D6A0((__int64)v40, *v40 + 17LL, 8);
              v37 = v62;
            }
            v39 += 8;
          }
          while ( v38 != v39 );
        }
        _libc_free(v37);
        if ( (_DWORD)v61 )
        {
          v41 = v59;
          if ( v60 )
          {
            v42 = 8LL * v60;
            v43 = 0;
            do
            {
              v44 = *(_QWORD **)(v41 + v43);
              if ( v44 && v44 != (_QWORD *)-8LL )
              {
                sub_C7D6A0((__int64)v44, *v44 + 17LL, 8);
                v41 = v59;
              }
              v43 += 8;
            }
            while ( v43 != v42 );
          }
        }
        else
        {
          v41 = v59;
        }
        _libc_free(v41);
        return 1;
      }
      a3[1] = 8;
    }
    if ( v22 )
      (*(void (__fastcall **)(_QWORD *))(*v22 + 8LL))(v22);
    goto LABEL_57;
  }
  if ( *(_QWORD *)v5 ^ 0x6F6C2D787470766ELL | *(_QWORD *)(v5 + 8) ^ 0x726F74632D726577LL
    || *(_DWORD *)(v5 + 16) != 1869898797
    || *(_BYTE *)(v5 + 20) != 114 )
  {
    return 0;
  }
  v7 = sub_22077B0(0x10u);
  if ( v7 )
    *(_QWORD *)v7 = &unk_4A30F20;
LABEL_15:
  v62 = v7;
  v10 = (char *)a3[1];
  if ( v10 == (char *)a3[2] )
  {
    sub_2275C60(a3, v10, &v62);
    v7 = v62;
LABEL_39:
    if ( v7 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
    return 1;
  }
  if ( !v10 )
  {
    a3[1] = 8;
    goto LABEL_39;
  }
  *(_QWORD *)v10 = v7;
  a3[1] += 8LL;
  return 1;
}
