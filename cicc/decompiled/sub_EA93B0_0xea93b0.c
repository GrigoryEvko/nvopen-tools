// Function: sub_EA93B0
// Address: 0xea93b0
//
__int64 __fastcall sub_EA93B0(__int64 *a1, __int64 *a2, __int64 *a3, _QWORD *a4)
{
  char *v8; // r14
  char *v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // r11
  __int64 v16; // r8
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rcx
  _QWORD *v20; // r13
  __int64 v21; // rdx
  __int64 *v22; // rdx
  __int64 result; // rax
  __int64 v24; // rdx
  _QWORD *v25; // r15
  __int64 v26; // r14
  __int64 v27; // r12
  __int64 v28; // rdi
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
  __int64 v39; // rax
  __int64 v40; // rax
  char *v41; // r14
  size_t v42; // rdx
  char *v43; // rax
  __int64 v44; // [rsp+8h] [rbp-48h]
  __int64 v45; // [rsp+10h] [rbp-40h]
  __int64 v46; // [rsp+10h] [rbp-40h]
  char *v47; // [rsp+10h] [rbp-40h]
  __int64 v48; // [rsp+10h] [rbp-40h]
  _QWORD *v49; // [rsp+18h] [rbp-38h]
  __int64 v50; // [rsp+18h] [rbp-38h]
  __int64 v51; // [rsp+18h] [rbp-38h]
  __int64 v52; // [rsp+18h] [rbp-38h]
  char *v53; // [rsp+18h] [rbp-38h]

  v8 = (char *)a1[9];
  v9 = (char *)a1[5];
  v10 = v8 - v9;
  v11 = 0x2E8BA2E8BA2E8BA3LL * ((a1[6] - a1[7]) >> 3);
  v12 = (v8 - v9) >> 3;
  if ( 5 * v12 - 5 + v11 + 0x2E8BA2E8BA2E8BA3LL * ((a1[4] - a1[2]) >> 3) == 0x1745D1745D1745DLL )
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
          v48 = v10;
          v53 = v38;
          memmove(&v38[v10 + 8 - v42], v9, v42);
          v38 = v53;
          v10 = v48;
        }
      }
      else if ( v9 != v41 )
      {
        v52 = v10;
        v43 = (char *)memmove(v38, v9, v42);
        v10 = v52;
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
      v50 = v10;
      v31 = sub_22077B0(8 * v30);
      v32 = (const void *)a1[5];
      v33 = v31;
      v34 = v50;
      v35 = (void *)(v31 + 8 * ((v30 - (v12 + 2)) >> 1));
      v36 = a1[9] + 8;
      if ( (const void *)v36 != v32 )
      {
        v46 = v33;
        v37 = memmove(v35, v32, v36 - (_QWORD)v32);
        v33 = v46;
        v34 = v50;
        v35 = v37;
      }
      v44 = v33;
      v47 = (char *)v35;
      v51 = v34;
      j_j___libc_free_0(*a1, 8 * a1[1]);
      a1[1] = v30;
      v38 = v47;
      v10 = v51;
      *a1 = v44;
    }
    a1[5] = (__int64)v38;
    v39 = *(_QWORD *)v38;
    v8 = &v38[v10];
    a1[9] = (__int64)&v38[v10];
    a1[3] = v39;
    a1[4] = v39 + 440;
    v40 = *(_QWORD *)&v38[v10];
    a1[7] = v40;
    a1[8] = v40 + 440;
  }
  *((_QWORD *)v8 + 1) = sub_22077B0(440);
  v14 = a1[6];
  v15 = (_QWORD *)*a4;
  v16 = *a2;
  v17 = a2[1];
  v18 = *a3;
  v19 = a3[1];
  v20 = (_QWORD *)a4[1];
  v49 = (_QWORD *)*a4;
  v21 = a4[2];
  a4[1] = 0;
  a4[2] = 0;
  *a4 = 0;
  if ( v14 )
  {
    *(_QWORD *)v14 = v16;
    *(_QWORD *)(v14 + 8) = v17;
    *(_QWORD *)(v14 + 16) = v18;
    *(_QWORD *)(v14 + 24) = v19;
    *(_QWORD *)(v14 + 32) = v15;
    *(_QWORD *)(v14 + 40) = v20;
    *(_QWORD *)(v14 + 48) = v21;
    *(_QWORD *)(v14 + 56) = 0;
    *(_QWORD *)(v14 + 64) = 0;
    *(_QWORD *)(v14 + 72) = 0;
    *(_BYTE *)(v14 + 80) = 0;
    *(_DWORD *)(v14 + 84) = 0;
  }
  else
  {
    v45 = v21 - (_QWORD)v49;
    if ( v49 != v20 )
    {
      v25 = v49;
      do
      {
        v26 = v25[3];
        v27 = v25[2];
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
            v27 += 40;
          }
          while ( v26 != v27 );
          v27 = v25[2];
        }
        if ( v27 )
          j_j___libc_free_0(v27, v25[4] - v27);
        v25 += 6;
      }
      while ( v20 != v25 );
    }
    if ( v49 )
      j_j___libc_free_0(v49, v45);
  }
  v22 = (__int64 *)(a1[9] + 8);
  a1[9] = (__int64)v22;
  result = *v22;
  v24 = *v22 + 440;
  a1[7] = result;
  a1[8] = v24;
  a1[6] = result;
  return result;
}
