// Function: sub_20F7DC0
// Address: 0x20f7dc0
//
char **__fastcall sub_20F7DC0(unsigned int *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  unsigned int v15; // edx
  _WORD *v16; // rcx
  int v17; // r9d
  unsigned __int16 *v18; // rdx
  unsigned __int16 v19; // r15
  char **result; // rax
  unsigned __int16 *v21; // r14
  __int64 v22; // rdx
  int *v23; // rax
  int v24; // ecx
  unsigned int v25; // eax
  __int64 v26; // r12
  __int64 v27; // rax
  char *v28; // rdi
  _QWORD *v29; // r12
  __int64 v30; // rcx
  _QWORD *v31; // rax
  __int64 v32; // r8
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 i; // rdx
  __int64 v38; // [rsp+8h] [rbp-C8h]
  __int64 v39; // [rsp+8h] [rbp-C8h]
  _QWORD *v40; // [rsp+8h] [rbp-C8h]
  char v41; // [rsp+10h] [rbp-C0h]
  __int64 v42; // [rsp+10h] [rbp-C0h]
  __int64 v43; // [rsp+10h] [rbp-C0h]
  _DWORD *v45; // [rsp+30h] [rbp-A0h]
  char *v46; // [rsp+38h] [rbp-98h] BYREF
  __int64 v47; // [rsp+40h] [rbp-90h]
  _BYTE v48[64]; // [rsp+48h] [rbp-88h] BYREF
  int v49; // [rsp+88h] [rbp-48h]
  __int64 v50; // [rsp+90h] [rbp-40h]
  __int64 v51; // [rsp+98h] [rbp-38h]

  *a1 = a2;
  v8 = a1[130];
  ++a1[1];
  v9 = (__int64)(*(_QWORD *)(a5 + 104) - *(_QWORD *)(a5 + 96)) >> 3;
  if ( (unsigned int)v9 < v8 )
  {
LABEL_35:
    a1[130] = v9;
    goto LABEL_3;
  }
  if ( (unsigned int)v9 > v8 )
  {
    if ( (unsigned int)v9 > (unsigned __int64)a1[131] )
    {
      sub_16CD150((__int64)(a1 + 128), a1 + 132, (unsigned int)v9, 24, a5, a6);
      v8 = a1[130];
    }
    v35 = *((_QWORD *)a1 + 64);
    v36 = v35 + 24 * v8;
    for ( i = v35 + 24LL * (unsigned int)v9; i != v36; v36 += 24 )
    {
      if ( v36 )
      {
        *(_DWORD *)v36 = 0;
        *(_QWORD *)(v36 + 8) = 0;
        *(_QWORD *)(v36 + 16) = 0;
      }
    }
    goto LABEL_35;
  }
LABEL_3:
  v10 = a1[14];
  v11 = *((_QWORD *)a1 + 6);
  *((_QWORD *)a1 + 5) = 0;
  v12 = v11 + 112 * v10;
  while ( v11 != v12 )
  {
    while ( 1 )
    {
      v12 -= 112;
      v13 = *(_QWORD *)(v12 + 8);
      if ( v13 == v12 + 24 )
        break;
      _libc_free(v13);
      if ( v11 == v12 )
        goto LABEL_7;
    }
  }
LABEL_7:
  a1[14] = 0;
  if ( !a4 )
    BUG();
  v14 = *a1;
  v15 = *(_DWORD *)(*(_QWORD *)(a4 + 8) + 24 * v14 + 16);
  v17 = v14 * (v15 & 0xF);
  v16 = (_WORD *)(*(_QWORD *)(a4 + 56) + 2LL * (v15 >> 4));
  LOWORD(v17) = *v16 + v14 * (v15 & 0xF);
  v18 = v16 + 1;
  v19 = v17;
  while ( 1 )
  {
    result = &v46;
    v21 = v18;
    if ( !v18 )
      break;
    while ( 1 )
    {
      v22 = 0x400000000LL;
      v50 = 0;
      v47 = 0x400000000LL;
      v23 = (int *)(a3 + 216LL * v19);
      v24 = *v23;
      v45 = v23 + 2;
      v25 = a1[14];
      v46 = v48;
      v49 = v24;
      if ( v25 >= a1[15] )
      {
        sub_20F7BE0((__int64)(a1 + 12), 0);
        v25 = a1[14];
      }
      v26 = *((_QWORD *)a1 + 6) + 112LL * v25;
      if ( v26 )
      {
        *(_QWORD *)v26 = v45;
        *(_QWORD *)(v26 + 8) = v26 + 24;
        *(_QWORD *)(v26 + 16) = 0x400000000LL;
        if ( (_DWORD)v47 )
          sub_20F7880(v26 + 8, &v46, v22, v25, a5, v17);
        *(_DWORD *)(v26 + 88) = v49;
        *(_QWORD *)(v26 + 96) = v50;
        *(_QWORD *)(v26 + 104) = v51;
        v25 = a1[14];
      }
      v27 = v25 + 1;
      v28 = v46;
      a1[14] = v27;
      if ( v28 != v48 )
      {
        _libc_free((unsigned __int64)v28);
        v27 = a1[14];
      }
      v29 = (_QWORD *)*((_QWORD *)a1 + 4);
      a5 = *((_QWORD *)a1 + 6) + 112 * v27 - 112;
      v30 = *(_QWORD *)(v29[84] + 8LL * v19);
      if ( !v30 )
      {
        v38 = *((_QWORD *)a1 + 6) + 112 * v27 - 112;
        v41 = qword_4FC4440[20];
        v31 = (_QWORD *)sub_22077B0(104);
        v32 = v38;
        v33 = (__int64)v31;
        if ( v31 )
        {
          *v31 = v31 + 2;
          v31[1] = 0x200000000LL;
          v31[8] = v31 + 10;
          v31[9] = 0x200000000LL;
          if ( v41 )
          {
            v40 = v31;
            v43 = v32;
            v34 = sub_22077B0(48);
            v32 = v43;
            v33 = (__int64)v40;
            if ( v34 )
            {
              *(_DWORD *)(v34 + 8) = 0;
              *(_QWORD *)(v34 + 16) = 0;
              *(_QWORD *)(v34 + 24) = v34 + 8;
              *(_QWORD *)(v34 + 32) = v34 + 8;
              *(_QWORD *)(v34 + 40) = 0;
            }
            v40[12] = v34;
          }
          else
          {
            v31[12] = 0;
          }
        }
        v39 = v32;
        *(_QWORD *)(v29[84] + 8LL * v19) = v33;
        v42 = v33;
        sub_1DBA8F0(v29, v33, v19);
        a5 = v39;
        v30 = v42;
      }
      *(_QWORD *)(a5 + 96) = v30;
      result = (char **)*v21;
      v18 = 0;
      ++v21;
      if ( !(_WORD)result )
        break;
      v19 += (unsigned __int16)result;
      if ( !v21 )
        return result;
    }
  }
  return result;
}
