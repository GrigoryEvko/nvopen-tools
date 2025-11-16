// Function: sub_2272560
// Address: 0x2272560
//
void __fastcall sub_2272560(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  __int64 *v6; // r14
  __int64 *v7; // r12
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 *v11; // r14
  __int64 *v12; // r12
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 *v16; // r14
  __int64 *v17; // r12
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // rdi
  __int64 *v21; // r14
  __int64 *v22; // r12
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rdi
  __int64 *v26; // r14
  __int64 *v27; // r12
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rdi
  __int64 *v31; // r14
  __int64 *v32; // r12
  __int64 v33; // rax
  __int64 v34; // r13
  __int64 v35; // rdi
  __int64 *v36; // r14
  __int64 *v37; // r12
  __int64 v38; // rax
  __int64 v39; // r13
  __int64 v40; // rdi
  __int64 *v41; // r14
  __int64 *v42; // r12
  __int64 v43; // rax
  __int64 v44; // r13
  __int64 v45; // rdi
  __int64 *v46; // r14
  __int64 *v47; // r12
  __int64 v48; // rax
  __int64 v49; // r13
  __int64 v50; // rdi
  __int64 *v51; // r14
  __int64 *v52; // r12
  __int64 v53; // rax
  __int64 v54; // r13
  __int64 v55; // rdi

  v1 = *(unsigned int *)(a1 + 1464);
  if ( (_DWORD)v1 )
  {
    v3 = *(_QWORD **)(a1 + 1448);
    v4 = &v3[6 * v1];
    do
    {
      while ( 1 )
      {
        if ( *v3 != -1 && *v3 != -2 )
        {
          v5 = v3[2];
          if ( (_QWORD *)v5 != v3 + 4 )
            break;
        }
        v3 += 6;
        if ( v4 == v3 )
          goto LABEL_8;
      }
      v3 += 6;
      j_j___libc_free_0(v5);
    }
    while ( v4 != v3 );
LABEL_8:
    v1 = *(unsigned int *)(a1 + 1464);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1448), 48 * v1, 8);
  v6 = *(__int64 **)(a1 + 1296);
  v7 = &v6[4 * *(unsigned int *)(a1 + 1304)];
  if ( v6 != v7 )
  {
    do
    {
      v8 = *(v7 - 1);
      v7 -= 4;
      if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v9 = (v8 >> 1) & 1;
        if ( (v8 & 4) != 0 )
        {
          v10 = (__int64)v7;
          if ( !(_BYTE)v9 )
            v10 = *v7;
          (*(void (__fastcall **)(__int64))((v8 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v10);
        }
        if ( !(_BYTE)v9 )
          sub_C7D6A0(*v7, v7[1], v7[2]);
      }
    }
    while ( v6 != v7 );
    v7 = *(__int64 **)(a1 + 1296);
  }
  if ( v7 != (__int64 *)(a1 + 1312) )
    _libc_free((unsigned __int64)v7);
  v11 = *(__int64 **)(a1 + 1152);
  v12 = &v11[4 * *(unsigned int *)(a1 + 1160)];
  if ( v11 != v12 )
  {
    do
    {
      v13 = *(v12 - 1);
      v12 -= 4;
      if ( (v13 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v14 = (v13 >> 1) & 1;
        if ( (v13 & 4) != 0 )
        {
          v15 = (__int64)v12;
          if ( !(_BYTE)v14 )
            v15 = *v12;
          (*(void (__fastcall **)(__int64))((v13 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v15);
        }
        if ( !(_BYTE)v14 )
          sub_C7D6A0(*v12, v12[1], v12[2]);
      }
    }
    while ( v11 != v12 );
    v12 = *(__int64 **)(a1 + 1152);
  }
  if ( v12 != (__int64 *)(a1 + 1168) )
    _libc_free((unsigned __int64)v12);
  v16 = *(__int64 **)(a1 + 1008);
  v17 = &v16[4 * *(unsigned int *)(a1 + 1016)];
  if ( v16 != v17 )
  {
    do
    {
      v18 = *(v17 - 1);
      v17 -= 4;
      if ( (v18 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v19 = (v18 >> 1) & 1;
        if ( (v18 & 4) != 0 )
        {
          v20 = (__int64)v17;
          if ( !(_BYTE)v19 )
            v20 = *v17;
          (*(void (__fastcall **)(__int64))((v18 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v20);
        }
        if ( !(_BYTE)v19 )
          sub_C7D6A0(*v17, v17[1], v17[2]);
      }
    }
    while ( v16 != v17 );
    v17 = *(__int64 **)(a1 + 1008);
  }
  if ( v17 != (__int64 *)(a1 + 1024) )
    _libc_free((unsigned __int64)v17);
  v21 = *(__int64 **)(a1 + 864);
  v22 = &v21[4 * *(unsigned int *)(a1 + 872)];
  if ( v21 != v22 )
  {
    do
    {
      v23 = *(v22 - 1);
      v22 -= 4;
      if ( (v23 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v24 = (v23 >> 1) & 1;
        if ( (v23 & 4) != 0 )
        {
          v25 = (__int64)v22;
          if ( !(_BYTE)v24 )
            v25 = *v22;
          (*(void (__fastcall **)(__int64))((v23 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v25);
        }
        if ( !(_BYTE)v24 )
          sub_C7D6A0(*v22, v22[1], v22[2]);
      }
    }
    while ( v21 != v22 );
    v22 = *(__int64 **)(a1 + 864);
  }
  if ( v22 != (__int64 *)(a1 + 880) )
    _libc_free((unsigned __int64)v22);
  v26 = *(__int64 **)(a1 + 720);
  v27 = &v26[4 * *(unsigned int *)(a1 + 728)];
  if ( v26 != v27 )
  {
    do
    {
      v28 = *(v27 - 1);
      v27 -= 4;
      if ( (v28 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v29 = (v28 >> 1) & 1;
        if ( (v28 & 4) != 0 )
        {
          v30 = (__int64)v27;
          if ( !(_BYTE)v29 )
            v30 = *v27;
          (*(void (__fastcall **)(__int64))((v28 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v30);
        }
        if ( !(_BYTE)v29 )
          sub_C7D6A0(*v27, v27[1], v27[2]);
      }
    }
    while ( v26 != v27 );
    v27 = *(__int64 **)(a1 + 720);
  }
  if ( v27 != (__int64 *)(a1 + 736) )
    _libc_free((unsigned __int64)v27);
  v31 = *(__int64 **)(a1 + 576);
  v32 = &v31[4 * *(unsigned int *)(a1 + 584)];
  if ( v31 != v32 )
  {
    do
    {
      v33 = *(v32 - 1);
      v32 -= 4;
      if ( (v33 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v34 = (v33 >> 1) & 1;
        if ( (v33 & 4) != 0 )
        {
          v35 = (__int64)v32;
          if ( !(_BYTE)v34 )
            v35 = *v32;
          (*(void (__fastcall **)(__int64))((v33 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v35);
        }
        if ( !(_BYTE)v34 )
          sub_C7D6A0(*v32, v32[1], v32[2]);
      }
    }
    while ( v31 != v32 );
    v32 = *(__int64 **)(a1 + 576);
  }
  if ( v32 != (__int64 *)(a1 + 592) )
    _libc_free((unsigned __int64)v32);
  v36 = *(__int64 **)(a1 + 432);
  v37 = &v36[4 * *(unsigned int *)(a1 + 440)];
  if ( v36 != v37 )
  {
    do
    {
      v38 = *(v37 - 1);
      v37 -= 4;
      if ( (v38 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v39 = (v38 >> 1) & 1;
        if ( (v38 & 4) != 0 )
        {
          v40 = (__int64)v37;
          if ( !(_BYTE)v39 )
            v40 = *v37;
          (*(void (__fastcall **)(__int64))((v38 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v40);
        }
        if ( !(_BYTE)v39 )
          sub_C7D6A0(*v37, v37[1], v37[2]);
      }
    }
    while ( v36 != v37 );
    v37 = *(__int64 **)(a1 + 432);
  }
  if ( v37 != (__int64 *)(a1 + 448) )
    _libc_free((unsigned __int64)v37);
  v41 = *(__int64 **)(a1 + 288);
  v42 = &v41[4 * *(unsigned int *)(a1 + 296)];
  if ( v41 != v42 )
  {
    do
    {
      v43 = *(v42 - 1);
      v42 -= 4;
      if ( (v43 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v44 = (v43 >> 1) & 1;
        if ( (v43 & 4) != 0 )
        {
          v45 = (__int64)v42;
          if ( !(_BYTE)v44 )
            v45 = *v42;
          (*(void (__fastcall **)(__int64))((v43 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v45);
        }
        if ( !(_BYTE)v44 )
          sub_C7D6A0(*v42, v42[1], v42[2]);
      }
    }
    while ( v41 != v42 );
    v42 = *(__int64 **)(a1 + 288);
  }
  if ( v42 != (__int64 *)(a1 + 304) )
    _libc_free((unsigned __int64)v42);
  v46 = *(__int64 **)(a1 + 144);
  v47 = &v46[4 * *(unsigned int *)(a1 + 152)];
  if ( v46 != v47 )
  {
    do
    {
      v48 = *(v47 - 1);
      v47 -= 4;
      if ( (v48 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v49 = (v48 >> 1) & 1;
        if ( (v48 & 4) != 0 )
        {
          v50 = (__int64)v47;
          if ( !(_BYTE)v49 )
            v50 = *v47;
          (*(void (__fastcall **)(__int64))((v48 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v50);
        }
        if ( !(_BYTE)v49 )
          sub_C7D6A0(*v47, v47[1], v47[2]);
      }
    }
    while ( v46 != v47 );
    v47 = *(__int64 **)(a1 + 144);
  }
  if ( v47 != (__int64 *)(a1 + 160) )
    _libc_free((unsigned __int64)v47);
  v51 = *(__int64 **)a1;
  v52 = (__int64 *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
  if ( *(__int64 **)a1 != v52 )
  {
    do
    {
      v53 = *(v52 - 1);
      v52 -= 4;
      if ( (v53 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v54 = (v53 >> 1) & 1;
        if ( (v53 & 4) != 0 )
        {
          v55 = (__int64)v52;
          if ( !(_BYTE)v54 )
            v55 = *v52;
          (*(void (__fastcall **)(__int64))((v53 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v55);
        }
        if ( !(_BYTE)v54 )
          sub_C7D6A0(*v52, v52[1], v52[2]);
      }
    }
    while ( v51 != v52 );
    v52 = *(__int64 **)a1;
  }
  if ( v52 != (__int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v52);
}
