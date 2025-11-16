// Function: sub_CF22E0
// Address: 0xcf22e0
//
unsigned __int8 *__fastcall sub_CF22E0(unsigned __int8 *a1)
{
  unsigned __int8 ***v1; // rbx
  unsigned __int8 **v2; // rdi
  __int64 v3; // rax
  unsigned __int8 *v4; // r12
  int v5; // r10d
  unsigned __int8 **v6; // r9
  unsigned int v7; // ecx
  unsigned __int8 **v8; // rdx
  unsigned __int8 *v9; // rax
  unsigned __int8 *v10; // rbx
  unsigned int v11; // eax
  unsigned __int8 *v12; // rsi
  int v13; // edx
  int v14; // r10d
  unsigned __int8 **v15; // r8
  __int64 v16; // rdx
  unsigned int v17; // r15d
  __int64 v18; // rax
  unsigned __int8 *v19; // r13
  unsigned __int8 ***v20; // r14
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 *v23; // rbx
  __int64 *v24; // r13
  __int64 v25; // rdi
  int v27; // r8d
  unsigned int v28; // r15d
  unsigned __int8 **v29; // rdi
  unsigned __int8 *v30; // rcx
  unsigned __int8 *v31; // [rsp+8h] [rbp-A8h]
  __int64 v32; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v33; // [rsp+18h] [rbp-98h]
  __int64 v34; // [rsp+20h] [rbp-90h]
  __int64 v35; // [rsp+28h] [rbp-88h]
  __int64 v36; // [rsp+30h] [rbp-80h] BYREF
  __int64 v37; // [rsp+38h] [rbp-78h]
  unsigned __int8 **v38; // [rsp+40h] [rbp-70h]
  unsigned __int8 **v39; // [rsp+48h] [rbp-68h]
  _QWORD *v40; // [rsp+50h] [rbp-60h]
  unsigned __int64 v41; // [rsp+58h] [rbp-58h]
  __int64 v42; // [rsp+60h] [rbp-50h]
  unsigned __int8 **v43; // [rsp+68h] [rbp-48h]
  _QWORD *v44; // [rsp+70h] [rbp-40h]
  unsigned __int8 ***v45; // [rsp+78h] [rbp-38h]

  v32 = 0;
  v31 = sub_CEFC00(a1, (unsigned __int8 **)1);
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v42 = 0;
  v37 = 8;
  v36 = sub_22077B0(64);
  v1 = (unsigned __int8 ***)(v36 + 24);
  v41 = v36 + 24;
  v2 = (unsigned __int8 **)sub_22077B0(512);
  *(_QWORD *)(v36 + 24) = v2;
  v39 = v2;
  v40 = v2 + 64;
  v45 = v1;
  v43 = v2;
  v44 = v2 + 64;
  v38 = v2;
  if ( v2 )
  {
    v3 = (__int64)(v2 + 1);
    *v2 = v31;
    v42 = (__int64)(v2 + 1);
  }
  else
  {
    v42 = 8;
    v3 = 8;
  }
  v4 = 0;
  while ( 1 )
  {
    if ( v2 == (unsigned __int8 **)v3 )
    {
      v10 = (*(v1 - 1))[63];
      j_j___libc_free_0(v2, 512);
      v16 = (__int64)(*--v45 + 64);
      v43 = *v45;
      v44 = (_QWORD *)v16;
      v42 = (__int64)(v43 + 63);
    }
    else
    {
      v10 = *(unsigned __int8 **)(v3 - 8);
      v42 = v3 - 8;
    }
    if ( !(_DWORD)v35 )
    {
      ++v32;
LABEL_11:
      sub_BD14B0((__int64)&v32, 2 * v35);
      if ( !(_DWORD)v35 )
        goto LABEL_75;
      v11 = (v35 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v6 = (unsigned __int8 **)(v33 + 8LL * v11);
      v12 = *v6;
      v13 = v34 + 1;
      if ( v10 != *v6 )
      {
        v14 = 1;
        v15 = 0;
        while ( v12 != (unsigned __int8 *)-4096LL )
        {
          if ( !v15 && v12 == (unsigned __int8 *)-8192LL )
            v15 = v6;
          v11 = (v35 - 1) & (v14 + v11);
          v6 = (unsigned __int8 **)(v33 + 8LL * v11);
          v12 = *v6;
          if ( v10 == *v6 )
            goto LABEL_27;
          ++v14;
        }
        if ( v15 )
          v6 = v15;
      }
      goto LABEL_27;
    }
    v5 = 1;
    v6 = 0;
    v7 = (v35 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v8 = (unsigned __int8 **)(v33 + 8LL * v7);
    v9 = *v8;
    if ( v10 == *v8 )
      goto LABEL_5;
    while ( v9 != (unsigned __int8 *)-4096LL )
    {
      if ( v9 != (unsigned __int8 *)-8192LL || v6 )
        v8 = v6;
      v7 = (v35 - 1) & (v5 + v7);
      v9 = *(unsigned __int8 **)(v33 + 8LL * v7);
      if ( v10 == v9 )
        goto LABEL_5;
      ++v5;
      v6 = v8;
      v8 = (unsigned __int8 **)(v33 + 8LL * v7);
    }
    if ( !v6 )
      v6 = v8;
    ++v32;
    v13 = v34 + 1;
    if ( 4 * ((int)v34 + 1) >= (unsigned int)(3 * v35) )
      goto LABEL_11;
    if ( (int)v35 - HIDWORD(v34) - v13 <= (unsigned int)v35 >> 3 )
    {
      sub_BD14B0((__int64)&v32, v35);
      if ( !(_DWORD)v35 )
      {
LABEL_75:
        LODWORD(v34) = v34 + 1;
        BUG();
      }
      v27 = 1;
      v28 = (v35 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v6 = (unsigned __int8 **)(v33 + 8LL * v28);
      v13 = v34 + 1;
      v29 = 0;
      v30 = *v6;
      if ( v10 != *v6 )
      {
        while ( v30 != (unsigned __int8 *)-4096LL )
        {
          if ( v30 == (unsigned __int8 *)-8192LL && !v29 )
            v29 = v6;
          v28 = (v35 - 1) & (v27 + v28);
          v6 = (unsigned __int8 **)(v33 + 8LL * v28);
          v30 = *v6;
          if ( v10 == *v6 )
            goto LABEL_27;
          ++v27;
        }
        if ( v29 )
          v6 = v29;
      }
    }
LABEL_27:
    LODWORD(v34) = v13;
    if ( *v6 != (unsigned __int8 *)-4096LL )
      --HIDWORD(v34);
    *v6 = v10;
    if ( *v10 != 84 )
      break;
    if ( (*((_DWORD *)v10 + 1) & 0x7FFFFFF) != 0 )
    {
      v17 = 0;
      do
      {
        v19 = sub_CEFC00(*(unsigned __int8 **)(*((_QWORD *)v10 - 1) + 32LL * v17), (unsigned __int8 **)1);
        v18 = v42;
        if ( (_QWORD *)v42 == v44 - 1 )
        {
          v20 = v45;
          if ( ((v42 - (__int64)v43) >> 3)
             + ((((__int64)((__int64)v45 - v41) >> 3) - 1) << 6)
             + (unsigned __int8 **)v40
             - v38 == 0xFFFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
          if ( (unsigned __int64)(v37 - (((__int64)v45 - v36) >> 3)) <= 1 )
          {
            sub_CF2160(&v36, 1u, 0);
            v20 = v45;
          }
          v20[1] = (unsigned __int8 **)sub_22077B0(512);
          if ( v42 )
            *(_QWORD *)v42 = v19;
          v21 = (__int64)(*++v45 + 64);
          v43 = *v45;
          v44 = (_QWORD *)v21;
          v42 = (__int64)v43;
        }
        else
        {
          if ( v42 )
          {
            *(_QWORD *)v42 = v19;
            v18 = v42;
          }
          v42 = v18 + 8;
        }
        ++v17;
      }
      while ( v17 != (*((_DWORD *)v10 + 1) & 0x7FFFFFF) );
    }
LABEL_5:
    v3 = v42;
    if ( v38 == (unsigned __int8 **)v42 )
      goto LABEL_48;
LABEL_6:
    v2 = v43;
    v1 = v45;
  }
  if ( v4 )
  {
    if ( v10 != v4 )
    {
      v4 = v31;
      goto LABEL_50;
    }
    goto LABEL_5;
  }
  v3 = v42;
  v4 = v10;
  if ( v38 != (unsigned __int8 **)v42 )
    goto LABEL_6;
LABEL_48:
  if ( !v4 )
    v4 = v31;
LABEL_50:
  v22 = v36;
  if ( v36 )
  {
    v23 = (__int64 *)v41;
    v24 = (__int64 *)(v45 + 1);
    if ( (unsigned __int64)(v45 + 1) > v41 )
    {
      do
      {
        v25 = *v23++;
        j_j___libc_free_0(v25, 512);
      }
      while ( v24 > v23 );
      v22 = v36;
    }
    j_j___libc_free_0(v22, 8 * v37);
  }
  sub_C7D6A0(v33, 8LL * (unsigned int)v35, 8);
  return v4;
}
