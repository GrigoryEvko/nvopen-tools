// Function: sub_38E9C50
// Address: 0x38e9c50
//
unsigned __int64 __fastcall sub_38E9C50(unsigned __int64 *a1, __int64 *a2, __int64 *a3, unsigned __int64 *a4)
{
  char *v8; // r14
  char *v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rax
  unsigned __int64 v15; // r11
  __int64 v16; // r8
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rcx
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // rdx
  unsigned __int64 *v22; // rdx
  unsigned __int64 result; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // r15
  __int64 v26; // r14
  unsigned __int64 v27; // r12
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  unsigned __int64 v30; // r14
  __int64 v31; // rax
  const void *v32; // rsi
  __int64 v33; // r10
  __int64 v34; // rcx
  void *v35; // r8
  __int64 v36; // rax
  void *v37; // rax
  char *v38; // r8
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rax
  char *v41; // r14
  size_t v42; // rdx
  char *v43; // rax
  unsigned __int64 v44; // [rsp+8h] [rbp-48h]
  __int64 v45; // [rsp+10h] [rbp-40h]
  char *v46; // [rsp+10h] [rbp-40h]
  __int64 v47; // [rsp+10h] [rbp-40h]
  unsigned __int64 v48; // [rsp+18h] [rbp-38h]
  __int64 v49; // [rsp+18h] [rbp-38h]
  __int64 v50; // [rsp+18h] [rbp-38h]
  __int64 v51; // [rsp+18h] [rbp-38h]
  char *v52; // [rsp+18h] [rbp-38h]

  v8 = (char *)a1[9];
  v9 = (char *)a1[5];
  v10 = v8 - v9;
  v11 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(a1[6] - a1[7]) >> 3);
  v12 = (v8 - v9) >> 3;
  if ( 9 * v12 - 9 + v11 + 0x6DB6DB6DB6DB6DB7LL * ((__int64)(a1[4] - a1[2]) >> 3) == 0x249249249249249LL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  v13 = a1[1];
  if ( v13 - ((__int64)&v8[-*a1] >> 3) <= 1 )
  {
    if ( v13 > 2 * (v12 + 2) )
    {
      v41 = v8 + 8;
      v38 = (char *)(*a1 + 8 * ((v13 - (v12 + 2)) >> 1));
      v42 = v41 - v9;
      if ( v9 <= v38 )
      {
        if ( v9 != v41 )
        {
          v47 = v10;
          v52 = v38;
          memmove(&v38[v10 + 8 - v42], v9, v42);
          v38 = v52;
          v10 = v47;
        }
      }
      else if ( v9 != v41 )
      {
        v51 = v10;
        v43 = (char *)memmove(v38, v9, v42);
        v10 = v51;
        v38 = v43;
      }
    }
    else
    {
      v29 = 1;
      if ( v13 )
        v29 = a1[1];
      v30 = v13 + v29 + 2;
      if ( v30 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v12, v9, v13);
      v49 = v10;
      v31 = sub_22077B0(8 * v30);
      v32 = (const void *)a1[5];
      v33 = v31;
      v34 = v49;
      v35 = (void *)(v31 + 8 * ((v30 - (v12 + 2)) >> 1));
      v36 = a1[9] + 8;
      if ( (const void *)v36 != v32 )
      {
        v45 = v33;
        v37 = memmove(v35, v32, v36 - (_QWORD)v32);
        v33 = v45;
        v34 = v49;
        v35 = v37;
      }
      v44 = v33;
      v46 = (char *)v35;
      v50 = v34;
      j_j___libc_free_0(*a1);
      a1[1] = v30;
      v38 = v46;
      v10 = v50;
      *a1 = v44;
    }
    a1[5] = (unsigned __int64)v38;
    v39 = *(_QWORD *)v38;
    v8 = &v38[v10];
    a1[9] = (unsigned __int64)&v38[v10];
    a1[3] = v39;
    a1[4] = v39 + 504;
    v40 = *(_QWORD *)&v38[v10];
    a1[7] = v40;
    a1[8] = v40 + 504;
  }
  *((_QWORD *)v8 + 1) = sub_22077B0(0x1F8u);
  v14 = (_QWORD *)a1[6];
  v15 = *a4;
  v16 = *a2;
  v17 = a2[1];
  v18 = *a3;
  v19 = a3[1];
  v20 = a4[1];
  v48 = *a4;
  v21 = a4[2];
  a4[1] = 0;
  a4[2] = 0;
  *a4 = 0;
  if ( v14 )
  {
    *v14 = v16;
    v14[1] = v17;
    v14[2] = v18;
    v14[3] = v19;
    v14[4] = v15;
    v14[5] = v20;
    v14[6] = v21;
  }
  else
  {
    if ( v48 != v20 )
    {
      v25 = v48;
      do
      {
        v26 = *(_QWORD *)(v25 + 24);
        v27 = *(_QWORD *)(v25 + 16);
        if ( v26 != v27 )
        {
          do
          {
            if ( *(_DWORD *)(v27 + 32) > 0x40u )
            {
              v28 = *(_QWORD *)(v27 + 24);
              if ( v28 )
                j_j___libc_free_0_0(v28);
            }
            v27 += 40LL;
          }
          while ( v26 != v27 );
          v27 = *(_QWORD *)(v25 + 16);
        }
        if ( v27 )
          j_j___libc_free_0(v27);
        v25 += 48LL;
      }
      while ( v20 != v25 );
    }
    if ( v48 )
      j_j___libc_free_0(v48);
  }
  v22 = (unsigned __int64 *)(a1[9] + 8);
  a1[9] = (unsigned __int64)v22;
  result = *v22;
  v24 = *v22 + 504;
  a1[7] = result;
  a1[8] = v24;
  a1[6] = result;
  return result;
}
