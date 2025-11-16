// Function: sub_38EAA00
// Address: 0x38eaa00
//
__int64 __fastcall sub_38EAA00(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rax
  unsigned __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r8
  unsigned __int64 v9; // rdi
  __int64 v10; // r15
  __int64 v11; // rbx
  unsigned __int64 v12; // r13
  __int64 v13; // r14
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  __int64 v16; // r15
  __int64 v17; // r14
  unsigned __int64 v18; // r12
  __int64 v19; // rbx
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // rdi
  __int64 v22; // r15
  __int64 v23; // r14
  unsigned __int64 v24; // r12
  __int64 v25; // rbx
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 *v29; // rbx
  unsigned __int64 v30; // r13
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // r13
  __int64 v37; // r8
  unsigned __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v41; // r14
  __int64 v42; // r15
  unsigned __int64 v43; // r12
  __int64 v44; // rbx
  unsigned __int64 v45; // r13
  unsigned __int64 v46; // rdi
  __int64 v47; // [rsp+8h] [rbp-78h]
  __int64 v48; // [rsp+10h] [rbp-70h]
  __int64 v49; // [rsp+18h] [rbp-68h]
  __int64 v50; // [rsp+20h] [rbp-60h]
  unsigned __int64 v51; // [rsp+28h] [rbp-58h]
  __int64 v52; // [rsp+30h] [rbp-50h]
  __int64 *v53; // [rsp+38h] [rbp-48h]
  __int64 v54; // [rsp+40h] [rbp-40h]
  __int64 v55; // [rsp+48h] [rbp-38h]
  __int64 v56; // [rsp+48h] [rbp-38h]
  __int64 v57; // [rsp+48h] [rbp-38h]

  v1 = a1;
  *(_QWORD *)a1 = off_49D9310;
  v2 = *(_QWORD *)(a1 + 344);
  v3 = *(_QWORD *)(a1 + 352);
  *(_QWORD *)(v2 + 56) = *(_QWORD *)(a1 + 360);
  *(_QWORD *)(v2 + 48) = v3;
  if ( *(_DWORD *)(a1 + 860) )
  {
    v4 = *(unsigned int *)(a1 + 856);
    v5 = *(_QWORD *)(a1 + 848);
    if ( (_DWORD)v4 )
    {
      v6 = 8 * v4;
      v7 = 0;
      do
      {
        v8 = *(_QWORD *)(v5 + v7);
        if ( v8 != -8 && v8 )
        {
          _libc_free(*(_QWORD *)(v5 + v7));
          v5 = *(_QWORD *)(v1 + 848);
        }
        v7 += 8;
      }
      while ( v6 != v7 );
    }
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 848);
  }
  _libc_free(v5);
  v9 = *(_QWORD *)(v1 + 600);
  if ( v9 != v1 + 616 )
    _libc_free(v9);
  v54 = *(_QWORD *)(v1 + 520);
  v52 = *(_QWORD *)(v1 + 488);
  v49 = *(_QWORD *)(v1 + 528);
  v50 = *(_QWORD *)(v1 + 504);
  v51 = *(_QWORD *)(v1 + 544);
  v48 = *(_QWORD *)(v1 + 512);
  v53 = (__int64 *)(v48 + 8);
  if ( v51 > v48 + 8 )
  {
    v47 = v1;
    do
    {
      v10 = *v53;
      v55 = *v53 + 504;
      do
      {
        v11 = *(_QWORD *)(v10 + 40);
        v12 = *(_QWORD *)(v10 + 32);
        if ( v11 != v12 )
        {
          do
          {
            v13 = *(_QWORD *)(v12 + 24);
            v14 = *(_QWORD *)(v12 + 16);
            if ( v13 != v14 )
            {
              do
              {
                if ( *(_DWORD *)(v14 + 32) > 0x40u )
                {
                  v15 = *(_QWORD *)(v14 + 24);
                  if ( v15 )
                    j_j___libc_free_0_0(v15);
                }
                v14 += 40LL;
              }
              while ( v13 != v14 );
              v14 = *(_QWORD *)(v12 + 16);
            }
            if ( v14 )
              j_j___libc_free_0(v14);
            v12 += 48LL;
          }
          while ( v11 != v12 );
          v12 = *(_QWORD *)(v10 + 32);
        }
        if ( v12 )
          j_j___libc_free_0(v12);
        v10 += 56;
      }
      while ( v55 != v10 );
      ++v53;
    }
    while ( v51 > (unsigned __int64)v53 );
    v1 = v47;
  }
  if ( v51 == v48 )
  {
    if ( v54 != v52 )
    {
      v57 = v1;
      v41 = v52;
      do
      {
        v42 = *(_QWORD *)(v41 + 40);
        v43 = *(_QWORD *)(v41 + 32);
        if ( v42 != v43 )
        {
          do
          {
            v44 = *(_QWORD *)(v43 + 24);
            v45 = *(_QWORD *)(v43 + 16);
            if ( v44 != v45 )
            {
              do
              {
                if ( *(_DWORD *)(v45 + 32) > 0x40u )
                {
                  v46 = *(_QWORD *)(v45 + 24);
                  if ( v46 )
                    j_j___libc_free_0_0(v46);
                }
                v45 += 40LL;
              }
              while ( v44 != v45 );
              v45 = *(_QWORD *)(v43 + 16);
            }
            if ( v45 )
              j_j___libc_free_0(v45);
            v43 += 48LL;
          }
          while ( v42 != v43 );
          v43 = *(_QWORD *)(v41 + 32);
        }
        if ( v43 )
          j_j___libc_free_0(v43);
        v41 += 56;
      }
      while ( v54 != v41 );
      goto LABEL_63;
    }
  }
  else
  {
    if ( v52 != v50 )
    {
      v56 = v1;
      v16 = v52;
      do
      {
        v17 = *(_QWORD *)(v16 + 40);
        v18 = *(_QWORD *)(v16 + 32);
        if ( v17 != v18 )
        {
          do
          {
            v19 = *(_QWORD *)(v18 + 24);
            v20 = *(_QWORD *)(v18 + 16);
            if ( v19 != v20 )
            {
              do
              {
                if ( *(_DWORD *)(v20 + 32) > 0x40u )
                {
                  v21 = *(_QWORD *)(v20 + 24);
                  if ( v21 )
                    j_j___libc_free_0_0(v21);
                }
                v20 += 40LL;
              }
              while ( v19 != v20 );
              v20 = *(_QWORD *)(v18 + 16);
            }
            if ( v20 )
              j_j___libc_free_0(v20);
            v18 += 48LL;
          }
          while ( v17 != v18 );
          v18 = *(_QWORD *)(v16 + 32);
        }
        if ( v18 )
          j_j___libc_free_0(v18);
        v16 += 56;
      }
      while ( v50 != v16 );
      v1 = v56;
    }
    if ( v54 != v49 )
    {
      v57 = v1;
      v22 = v49;
      do
      {
        v23 = *(_QWORD *)(v22 + 40);
        v24 = *(_QWORD *)(v22 + 32);
        if ( v23 != v24 )
        {
          do
          {
            v25 = *(_QWORD *)(v24 + 24);
            v26 = *(_QWORD *)(v24 + 16);
            if ( v25 != v26 )
            {
              do
              {
                if ( *(_DWORD *)(v26 + 32) > 0x40u )
                {
                  v27 = *(_QWORD *)(v26 + 24);
                  if ( v27 )
                    j_j___libc_free_0_0(v27);
                }
                v26 += 40LL;
              }
              while ( v25 != v26 );
              v26 = *(_QWORD *)(v24 + 16);
            }
            if ( v26 )
              j_j___libc_free_0(v26);
            v24 += 48LL;
          }
          while ( v23 != v24 );
          v24 = *(_QWORD *)(v22 + 32);
        }
        if ( v24 )
          j_j___libc_free_0(v24);
        v22 += 56;
      }
      while ( v54 != v22 );
LABEL_63:
      v1 = v57;
    }
  }
  v28 = *(_QWORD *)(v1 + 472);
  if ( v28 )
  {
    v29 = *(unsigned __int64 **)(v1 + 512);
    v30 = *(_QWORD *)(v1 + 544) + 8LL;
    if ( v30 > (unsigned __int64)v29 )
    {
      do
      {
        v31 = *v29++;
        j_j___libc_free_0(v31);
      }
      while ( v30 > (unsigned __int64)v29 );
      v28 = *(_QWORD *)(v1 + 472);
    }
    j_j___libc_free_0(v28);
  }
  v32 = *(_QWORD *)(v1 + 448);
  if ( v32 )
    j_j___libc_free_0(v32);
  v33 = *(_QWORD *)(v1 + 416);
  if ( *(_DWORD *)(v1 + 428) )
  {
    v34 = *(unsigned int *)(v1 + 424);
    if ( (_DWORD)v34 )
    {
      v35 = 8 * v34;
      v36 = 0;
      do
      {
        v37 = *(_QWORD *)(v33 + v36);
        if ( v37 && v37 != -8 )
        {
          _libc_free(*(_QWORD *)(v33 + v36));
          v33 = *(_QWORD *)(v1 + 416);
        }
        v36 += 8;
      }
      while ( v36 != v35 );
    }
  }
  _libc_free(v33);
  v38 = *(_QWORD *)(v1 + 392);
  if ( v38 )
    j_j___libc_free_0(v38);
  v39 = *(_QWORD *)(v1 + 368);
  if ( v39 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v39 + 8LL))(v39);
  sub_392A250(v1 + 144);
  return sub_39093B0(v1);
}
