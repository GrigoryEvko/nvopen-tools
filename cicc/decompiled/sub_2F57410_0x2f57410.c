// Function: sub_2F57410
// Address: 0x2f57410
//
__int64 __fastcall sub_2F57410(
        __int64 a1,
        unsigned __int16 a2,
        __int64 a3,
        __int64 a4,
        unsigned int *a5,
        unsigned int *a6)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  char v16; // al
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  _QWORD *v21; // rax
  _QWORD *v22; // rcx
  __int64 v23; // rax
  bool v24; // cf
  unsigned __int64 v25; // rax
  __int64 v26; // r14
  unsigned int v27; // r12d
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 *v30; // r13
  __int64 *v31; // r15
  unsigned int v32; // ebx
  __int64 v33; // rdi
  __int64 v34; // r12
  __int64 v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r8
  __int64 v39; // r9
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rax
  int v42; // r13d
  __int64 v43; // r14
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned __int64 v46; // r10
  int v47; // r13d
  __int64 v48; // rbx
  __int64 v49; // rbx
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rdx
  unsigned __int64 v53; // rdi
  int v54; // eax
  _QWORD *v55; // r12
  _QWORD *v56; // rbx
  unsigned __int64 v57; // rdi
  unsigned __int64 v58; // rdi
  __int64 v59; // rax
  unsigned __int64 *v60; // [rsp+8h] [rbp-88h]
  unsigned int v63; // [rsp+1Ch] [rbp-74h]
  unsigned int v65; // [rsp+28h] [rbp-68h]
  int v66; // [rsp+30h] [rbp-60h]
  int v67; // [rsp+30h] [rbp-60h]
  unsigned __int64 v69; // [rsp+48h] [rbp-48h] BYREF
  __int64 v70; // [rsp+50h] [rbp-40h] BYREF
  __int64 v71; // [rsp+58h] [rbp-38h]

  v6 = *a5;
  v60 = (unsigned __int64 *)a4;
  if ( (_DWORD)v6 != 32 )
    goto LABEL_2;
  v63 = 0;
  v26 = 0;
  v27 = 0;
  v65 = -1;
  v66 = *a6;
  do
  {
    if ( v66 != v27 )
    {
      a4 = a1;
      v28 = v26 + *(_QWORD *)(a1 + 24176);
      if ( *(_DWORD *)v28 )
      {
        v29 = *(__int64 **)(v28 + 24);
        v30 = &v29[*(unsigned int *)(v28 + 32)];
        if ( v30 == v29 )
        {
          v32 = 0;
        }
        else
        {
          v31 = *(__int64 **)(v28 + 24);
          v32 = 0;
          do
          {
            v33 = *v31++;
            v32 += sub_39FAC40(v33);
          }
          while ( v30 != v31 );
        }
        if ( v32 < v65 )
        {
          v63 = v27;
          v65 = v32;
        }
      }
    }
    ++v27;
    v26 += 144;
  }
  while ( v27 != 32 );
  *a5 = 31;
  v34 = *(_QWORD *)(a1 + 24176);
  v35 = v34 + 144LL * v63;
  *(_DWORD *)v35 = *(_DWORD *)(v34 + 4464);
  v36 = *(_QWORD *)(v35 + 8);
  *(_DWORD *)(v35 + 4) = *(_DWORD *)(v34 + 4468);
  v37 = *(_QWORD *)(v34 + 4472);
  *(_QWORD *)(v35 + 16) = 0;
  if ( v36 )
    --*(_DWORD *)(v36 + 8);
  *(_QWORD *)(v35 + 8) = v37;
  if ( v37 )
    ++*(_DWORD *)(v37 + 8);
  sub_2F4CBD0(v35 + 24, v34 + 4488, v36, a4, (__int64)a5, (__int64)a6);
  *(_DWORD *)(v35 + 88) = *(_DWORD *)(v34 + 4552);
  if ( v35 + 96 != v34 + 4560 )
  {
    v40 = *(unsigned int *)(v34 + 4568);
    v41 = *(unsigned int *)(v35 + 104);
    v42 = *(_DWORD *)(v34 + 4568);
    if ( v40 <= v41 )
    {
      if ( *(_DWORD *)(v34 + 4568) )
        memmove(*(void **)(v35 + 96), *(const void **)(v34 + 4560), 4 * v40);
    }
    else
    {
      if ( v40 > *(unsigned int *)(v35 + 108) )
      {
        *(_DWORD *)(v35 + 104) = 0;
        v43 = 0;
        sub_C8D5F0(v35 + 96, (const void *)(v35 + 112), v40, 4u, v38, v39);
        v40 = *(unsigned int *)(v34 + 4568);
      }
      else
      {
        v43 = 4 * v41;
        if ( *(_DWORD *)(v35 + 104) )
        {
          memmove(*(void **)(v35 + 96), *(const void **)(v34 + 4560), 4 * v41);
          v40 = *(unsigned int *)(v34 + 4568);
        }
      }
      v44 = *(_QWORD *)(v34 + 4560);
      v45 = 4 * v40;
      if ( v44 + v43 != v45 + v44 )
        memcpy((void *)(v43 + *(_QWORD *)(v35 + 96)), (const void *)(v44 + v43), v45 - v43);
    }
    *(_DWORD *)(v35 + 104) = v42;
  }
  v6 = *a5;
  if ( *a6 != (_DWORD)v6 )
  {
LABEL_2:
    v7 = a1;
    v8 = *(unsigned int *)(a1 + 24184);
    if ( (unsigned int)v8 <= (unsigned int)v6 )
      goto LABEL_44;
LABEL_3:
    v9 = *(_QWORD *)(v7 + 24176);
    goto LABEL_4;
  }
  *a6 = v63;
  v7 = a1;
  v6 = *a5;
  v8 = *(unsigned int *)(a1 + 24184);
  if ( (unsigned int)v8 > (unsigned int)v6 )
    goto LABEL_3;
LABEL_44:
  v46 = (unsigned int)(v6 + 1);
  v47 = v6 + 1;
  if ( v46 == v8 )
  {
    v7 = a1;
    goto LABEL_3;
  }
  v48 = 144 * v46;
  if ( v46 < v8 )
  {
    v9 = *(_QWORD *)(a1 + 24176);
    v55 = (_QWORD *)(v9 + 144 * v8);
    v56 = (_QWORD *)(v9 + v48);
    if ( v55 != v56 )
    {
      do
      {
        v55 -= 18;
        v57 = v55[12];
        if ( (_QWORD *)v57 != v55 + 14 )
          _libc_free(v57);
        v58 = v55[3];
        if ( (_QWORD *)v58 != v55 + 5 )
          _libc_free(v58);
        v55[2] = 0;
        v59 = v55[1];
        if ( v59 )
          --*(_DWORD *)(v59 + 8);
      }
      while ( v56 != v55 );
      v9 = *(_QWORD *)(a1 + 24176);
    }
    *(_DWORD *)(a1 + 24184) = v47;
    v6 = *a5;
  }
  else
  {
    if ( v46 > *(unsigned int *)(a1 + 24188) )
    {
      v9 = sub_C8D7D0(a1 + 24176, a1 + 24192, v46, 0x90u, (unsigned __int64 *)&v70, a1 + 24192);
      sub_2F57150(a1 + 24176, v9);
      v53 = *(_QWORD *)(a1 + 24176);
      v54 = v70;
      if ( a1 + 24192 != v53 )
      {
        v67 = v70;
        _libc_free(v53);
        v54 = v67;
      }
      *(_QWORD *)(a1 + 24176) = v9;
      v8 = *(unsigned int *)(a1 + 24184);
      *(_DWORD *)(a1 + 24188) = v54;
    }
    else
    {
      v9 = *(_QWORD *)(a1 + 24176);
    }
    v49 = v9 + v48;
    v50 = v9 + 144 * v8;
    if ( v50 != v49 )
    {
      do
      {
        if ( v50 )
        {
          memset((void *)v50, 0, 0x90u);
          *(_DWORD *)(v50 + 36) = 6;
          *(_QWORD *)(v50 + 24) = v50 + 40;
          *(_QWORD *)(v50 + 96) = v50 + 112;
          *(_DWORD *)(v50 + 108) = 8;
        }
        v50 += 144;
      }
      while ( v49 != v50 );
      v9 = *(_QWORD *)(a1 + 24176);
    }
    *(_DWORD *)(a1 + 24184) = v47;
    v6 = *a5;
  }
LABEL_4:
  v10 = v9 + 144 * v6;
  v11 = *(_QWORD *)(v10 + 8);
  *(_QWORD *)v10 = a2;
  *(_QWORD *)(v10 + 16) = 0;
  if ( v11 )
    --*(_DWORD *)(v11 + 8);
  *(_QWORD *)(v10 + 8) = 0;
  if ( a2 )
  {
    v51 = sub_3502D00(a1 + 1008);
    v52 = *(_QWORD *)(v10 + 8);
    *(_QWORD *)(v10 + 16) = 0;
    if ( v52 )
      --*(_DWORD *)(v52 + 8);
    *(_QWORD *)(v10 + 8) = v51;
    if ( v51 )
      ++*(_DWORD *)(v51 + 8);
  }
  *(_DWORD *)(v10 + 88) = 0;
  *(_DWORD *)(v10 + 32) = 0;
  *(_DWORD *)(v10 + 104) = 0;
  sub_2FB0010(*(_QWORD *)(a1 + 832), v10 + 24);
  v71 = 0;
  v15 = *(_QWORD *)(v10 + 8);
  v69 = 0;
  v70 = v15;
  if ( v15 )
    ++*(_DWORD *)(v15 + 8);
  v16 = sub_2F51220((_QWORD *)a1, &v70, (__int64 *)&v69, v12, v13, v14);
  v19 = v70;
  v71 = 0;
  if ( v70 )
    --*(_DWORD *)(v70 + 8);
  if ( v16 )
  {
    if ( v69 < *v60 )
    {
      if ( (unsigned __int8)sub_2F51D00(a1, v10, v19, v17, v18) )
      {
        sub_2FB0100(*(_QWORD *)(a1 + 832));
        v21 = sub_2F4C690(*(_QWORD **)(v10 + 24), *(_QWORD *)(v10 + 24) + 8LL * *(unsigned int *)(v10 + 32));
        if ( v22 != v21 )
        {
          v23 = sub_2F52670((_QWORD *)a1, (_QWORD *)v10);
          v24 = __CFADD__(v69, v23);
          v25 = v69 + v23;
          if ( !v24 && v25 < *v60 )
          {
            *a6 = *a5;
            *v60 = v25;
          }
          ++*a5;
        }
      }
    }
  }
  return *a6;
}
