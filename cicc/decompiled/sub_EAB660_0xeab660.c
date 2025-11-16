// Function: sub_EAB660
// Address: 0xeab660
//
__int64 __fastcall sub_EAB660(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r13
  __int64 v8; // rbx
  _QWORD *v9; // r8
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r13
  __int64 v13; // rbx
  _QWORD *v14; // r8
  __int64 v15; // rdi
  __int64 v16; // rdi
  _QWORD *v17; // r15
  _QWORD *v18; // rbx
  _QWORD *v19; // r12
  _QWORD *v20; // rbx
  _QWORD *v21; // r14
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 v24; // rdi
  _QWORD *v25; // r14
  _QWORD *v26; // rbx
  _QWORD *v27; // r13
  _QWORD *v28; // r15
  _QWORD *v29; // r12
  __int64 v30; // rbx
  __int64 v31; // r13
  __int64 v32; // rdi
  _QWORD *v33; // r14
  _QWORD *v34; // rbx
  _QWORD *v35; // r13
  _QWORD *v36; // r15
  _QWORD *v37; // r12
  __int64 v38; // rbx
  __int64 v39; // r13
  __int64 v40; // rdi
  __int64 v41; // rdi
  __int64 *v42; // rbx
  unsigned __int64 v43; // r13
  __int64 v44; // rdi
  __int64 v45; // rdi
  __int64 v46; // r8
  __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // rbx
  _QWORD *v50; // rdi
  __int64 v51; // rdi
  __int64 v52; // rdi
  _QWORD *v54; // r14
  _QWORD *v55; // rbx
  _QWORD *v56; // r13
  _QWORD *v57; // r15
  _QWORD *v58; // r12
  __int64 v59; // rbx
  __int64 v60; // r13
  __int64 v61; // rdi
  __int64 v62; // [rsp+8h] [rbp-78h]
  __int64 v63; // [rsp+10h] [rbp-70h]
  _QWORD *v64; // [rsp+18h] [rbp-68h]
  _QWORD *v65; // [rsp+20h] [rbp-60h]
  unsigned __int64 v66; // [rsp+28h] [rbp-58h]
  _QWORD *v67; // [rsp+30h] [rbp-50h]
  _QWORD *v68; // [rsp+38h] [rbp-48h]
  _QWORD *v69; // [rsp+40h] [rbp-40h]
  __int64 v70; // [rsp+48h] [rbp-38h]
  __int64 v71; // [rsp+48h] [rbp-38h]
  __int64 v72; // [rsp+48h] [rbp-38h]

  v2 = a1;
  *(_QWORD *)a1 = off_49E47A8;
  *(_QWORD *)(*(_QWORD *)(a1 + 232) + 264LL) = 0;
  v3 = *(_QWORD *)(a1 + 248);
  v4 = *(_QWORD *)(a1 + 264);
  *(_QWORD *)(v3 + 48) = *(_QWORD *)(a1 + 256);
  *(_QWORD *)(v3 + 56) = v4;
  if ( *(_DWORD *)(a1 + 908) )
  {
    v5 = *(unsigned int *)(a1 + 904);
    v6 = *(_QWORD *)(a1 + 896);
    if ( (_DWORD)v5 )
    {
      v7 = 8 * v5;
      v8 = 0;
      do
      {
        v9 = *(_QWORD **)(v6 + v8);
        if ( v9 != (_QWORD *)-8LL && v9 )
        {
          a2 = *v9 + 17LL;
          sub_C7D6A0(*(_QWORD *)(v6 + v8), a2, 8);
          v6 = *(_QWORD *)(v2 + 896);
        }
        v8 += 8;
      }
      while ( v7 != v8 );
    }
  }
  else
  {
    v6 = *(_QWORD *)(a1 + 896);
  }
  _libc_free(v6, a2);
  if ( *(_DWORD *)(v2 + 884) )
  {
    v10 = *(unsigned int *)(v2 + 880);
    v11 = *(_QWORD *)(v2 + 872);
    if ( (_DWORD)v10 )
    {
      v12 = 8 * v10;
      v13 = 0;
      do
      {
        v14 = *(_QWORD **)(v11 + v13);
        if ( v14 && v14 != (_QWORD *)-8LL )
        {
          a2 = *v14 + 17LL;
          sub_C7D6A0(*(_QWORD *)(v11 + v13), a2, 8);
          v11 = *(_QWORD *)(v2 + 872);
        }
        v13 += 8;
      }
      while ( v12 != v13 );
    }
  }
  else
  {
    v11 = *(_QWORD *)(v2 + 872);
  }
  _libc_free(v11, a2);
  sub_EA2BF0(*(_QWORD *)(v2 + 832));
  v15 = *(_QWORD *)(v2 + 768);
  if ( v15 != v2 + 784 )
    _libc_free(v15, a2);
  v16 = *(_QWORD *)(v2 + 528);
  if ( v16 != v2 + 544 )
    _libc_free(v16, a2);
  v67 = *(_QWORD **)(v2 + 408);
  v69 = *(_QWORD **)(v2 + 440);
  v65 = *(_QWORD **)(v2 + 424);
  v64 = *(_QWORD **)(v2 + 448);
  v63 = *(_QWORD *)(v2 + 432);
  v66 = *(_QWORD *)(v2 + 464);
  v68 = (_QWORD *)(v63 + 8);
  if ( v66 > v63 + 8 )
  {
    v62 = v2;
    do
    {
      v17 = (_QWORD *)*v68;
      v70 = *v68 + 440LL;
      do
      {
        v18 = (_QWORD *)v17[8];
        v19 = (_QWORD *)v17[7];
        if ( v18 != v19 )
        {
          do
          {
            if ( (_QWORD *)*v19 != v19 + 2 )
            {
              a2 = v19[2] + 1LL;
              j_j___libc_free_0(*v19, a2);
            }
            v19 += 4;
          }
          while ( v18 != v19 );
          v19 = (_QWORD *)v17[7];
        }
        if ( v19 )
        {
          a2 = v17[9] - (_QWORD)v19;
          j_j___libc_free_0(v19, a2);
        }
        v20 = (_QWORD *)v17[5];
        v21 = (_QWORD *)v17[4];
        if ( v20 != v21 )
        {
          do
          {
            v22 = v21[3];
            v23 = v21[2];
            if ( v22 != v23 )
            {
              do
              {
                if ( *(_DWORD *)(v23 + 32) > 0x40u )
                {
                  v24 = *(_QWORD *)(v23 + 24);
                  if ( v24 )
                    j_j___libc_free_0_0(v24);
                }
                v23 += 40;
              }
              while ( v22 != v23 );
              v23 = v21[2];
            }
            if ( v23 )
            {
              a2 = v21[4] - v23;
              j_j___libc_free_0(v23, a2);
            }
            v21 += 6;
          }
          while ( v20 != v21 );
          v21 = (_QWORD *)v17[4];
        }
        if ( v21 )
        {
          a2 = v17[6] - (_QWORD)v21;
          j_j___libc_free_0(v21, a2);
        }
        v17 += 11;
      }
      while ( (_QWORD *)v70 != v17 );
      ++v68;
    }
    while ( v66 > (unsigned __int64)v68 );
    v2 = v62;
  }
  if ( v66 == v63 )
  {
    if ( v69 != v67 )
    {
      v72 = v2;
      v54 = v67;
      do
      {
        v55 = (_QWORD *)v54[8];
        v56 = (_QWORD *)v54[7];
        if ( v55 != v56 )
        {
          do
          {
            if ( (_QWORD *)*v56 != v56 + 2 )
            {
              a2 = v56[2] + 1LL;
              j_j___libc_free_0(*v56, a2);
            }
            v56 += 4;
          }
          while ( v55 != v56 );
          v56 = (_QWORD *)v54[7];
        }
        if ( v56 )
        {
          a2 = v54[9] - (_QWORD)v56;
          j_j___libc_free_0(v56, a2);
        }
        v57 = (_QWORD *)v54[5];
        v58 = (_QWORD *)v54[4];
        if ( v57 != v58 )
        {
          do
          {
            v59 = v58[3];
            v60 = v58[2];
            if ( v59 != v60 )
            {
              do
              {
                if ( *(_DWORD *)(v60 + 32) > 0x40u )
                {
                  v61 = *(_QWORD *)(v60 + 24);
                  if ( v61 )
                    j_j___libc_free_0_0(v61);
                }
                v60 += 40;
              }
              while ( v59 != v60 );
              v60 = v58[2];
            }
            if ( v60 )
            {
              a2 = v58[4] - v60;
              j_j___libc_free_0(v60, a2);
            }
            v58 += 6;
          }
          while ( v57 != v58 );
          v58 = (_QWORD *)v54[4];
        }
        if ( v58 )
        {
          a2 = v54[6] - (_QWORD)v58;
          j_j___libc_free_0(v58, a2);
        }
        v54 += 11;
      }
      while ( v69 != v54 );
      goto LABEL_93;
    }
  }
  else
  {
    if ( v67 != v65 )
    {
      v71 = v2;
      v25 = v67;
      do
      {
        v26 = (_QWORD *)v25[8];
        v27 = (_QWORD *)v25[7];
        if ( v26 != v27 )
        {
          do
          {
            if ( (_QWORD *)*v27 != v27 + 2 )
            {
              a2 = v27[2] + 1LL;
              j_j___libc_free_0(*v27, a2);
            }
            v27 += 4;
          }
          while ( v26 != v27 );
          v27 = (_QWORD *)v25[7];
        }
        if ( v27 )
        {
          a2 = v25[9] - (_QWORD)v27;
          j_j___libc_free_0(v27, a2);
        }
        v28 = (_QWORD *)v25[5];
        v29 = (_QWORD *)v25[4];
        if ( v28 != v29 )
        {
          do
          {
            v30 = v29[3];
            v31 = v29[2];
            if ( v30 != v31 )
            {
              do
              {
                if ( *(_DWORD *)(v31 + 32) > 0x40u )
                {
                  v32 = *(_QWORD *)(v31 + 24);
                  if ( v32 )
                    j_j___libc_free_0_0(v32);
                }
                v31 += 40;
              }
              while ( v30 != v31 );
              v31 = v29[2];
            }
            if ( v31 )
            {
              a2 = v29[4] - v31;
              j_j___libc_free_0(v31, a2);
            }
            v29 += 6;
          }
          while ( v28 != v29 );
          v29 = (_QWORD *)v25[4];
        }
        if ( v29 )
        {
          a2 = v25[6] - (_QWORD)v29;
          j_j___libc_free_0(v29, a2);
        }
        v25 += 11;
      }
      while ( v65 != v25 );
      v2 = v71;
    }
    if ( v69 != v64 )
    {
      v72 = v2;
      v33 = v64;
      do
      {
        v34 = (_QWORD *)v33[8];
        v35 = (_QWORD *)v33[7];
        if ( v34 != v35 )
        {
          do
          {
            if ( (_QWORD *)*v35 != v35 + 2 )
            {
              a2 = v35[2] + 1LL;
              j_j___libc_free_0(*v35, a2);
            }
            v35 += 4;
          }
          while ( v34 != v35 );
          v35 = (_QWORD *)v33[7];
        }
        if ( v35 )
        {
          a2 = v33[9] - (_QWORD)v35;
          j_j___libc_free_0(v35, a2);
        }
        v36 = (_QWORD *)v33[5];
        v37 = (_QWORD *)v33[4];
        if ( v36 != v37 )
        {
          do
          {
            v38 = v37[3];
            v39 = v37[2];
            if ( v38 != v39 )
            {
              do
              {
                if ( *(_DWORD *)(v39 + 32) > 0x40u )
                {
                  v40 = *(_QWORD *)(v39 + 24);
                  if ( v40 )
                    j_j___libc_free_0_0(v40);
                }
                v39 += 40;
              }
              while ( v38 != v39 );
              v39 = v37[2];
            }
            if ( v39 )
            {
              a2 = v37[4] - v39;
              j_j___libc_free_0(v39, a2);
            }
            v37 += 6;
          }
          while ( v36 != v37 );
          v37 = (_QWORD *)v33[4];
        }
        if ( v37 )
        {
          a2 = v33[6] - (_QWORD)v37;
          j_j___libc_free_0(v37, a2);
        }
        v33 += 11;
      }
      while ( v69 != v33 );
LABEL_93:
      v2 = v72;
    }
  }
  v41 = *(_QWORD *)(v2 + 392);
  if ( v41 )
  {
    v42 = *(__int64 **)(v2 + 432);
    v43 = *(_QWORD *)(v2 + 464) + 8LL;
    if ( v43 > (unsigned __int64)v42 )
    {
      do
      {
        v44 = *v42++;
        j_j___libc_free_0(v44, 440);
      }
      while ( v43 > (unsigned __int64)v42 );
      v41 = *(_QWORD *)(v2 + 392);
    }
    a2 = 8LL * *(_QWORD *)(v2 + 400);
    j_j___libc_free_0(v41, a2);
  }
  v45 = *(_QWORD *)(v2 + 368);
  if ( v45 )
  {
    a2 = *(_QWORD *)(v2 + 384) - v45;
    j_j___libc_free_0(v45, a2);
  }
  v46 = *(_QWORD *)(v2 + 344);
  if ( *(_DWORD *)(v2 + 356) )
  {
    v47 = *(unsigned int *)(v2 + 352);
    if ( (_DWORD)v47 )
    {
      v48 = 8 * v47;
      v49 = 0;
      do
      {
        v50 = *(_QWORD **)(v46 + v49);
        if ( v50 && v50 != (_QWORD *)-8LL )
        {
          a2 = *v50 + 25LL;
          sub_C7D6A0((__int64)v50, a2, 8);
          v46 = *(_QWORD *)(v2 + 344);
        }
        v49 += 8;
      }
      while ( v48 != v49 );
    }
  }
  _libc_free(v46, a2);
  v51 = *(_QWORD *)(v2 + 320);
  if ( v51 )
    j_j___libc_free_0(v51, *(_QWORD *)(v2 + 336) - v51);
  v52 = *(_QWORD *)(v2 + 272);
  if ( v52 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v52 + 8LL))(v52);
  sub_1095430(v2 + 40);
  return sub_ECD700(v2);
}
