// Function: sub_108B640
// Address: 0x108b640
//
__int64 __fastcall sub_108B640(__int64 *a1, __int64 *a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  char *v9; // r14
  char *v10; // rsi
  __int64 v11; // r13
  __int64 v12; // rcx
  __int64 v13; // rdi
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r15
  __int64 v20; // rax
  unsigned __int64 v21; // r14
  _QWORD *v22; // r15
  const void *v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  char *v27; // r14
  size_t v28; // rdx
  __int64 v29; // [rsp+8h] [rbp-38h]

  v4 = a1[6];
  if ( v4 == a1[8] - 96 )
  {
    v9 = (char *)a1[9];
    v10 = (char *)a1[5];
    v11 = v9 - v10;
    v12 = (v9 - v10) >> 3;
    if ( 0xAAAAAAAAAAAAAAABLL * ((v4 - a1[7]) >> 5) + 5 * v12 - 5 - 0x5555555555555555LL * ((a1[4] - a1[2]) >> 5) == 0x155555555555555LL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    v13 = *a1;
    v14 = a1[1];
    if ( v14 - ((__int64)&v9[-*a1] >> 3) <= 1 )
    {
      v19 = v12 + 2;
      if ( v14 > 2 * (v12 + 2) )
      {
        v27 = v9 + 8;
        v22 = (_QWORD *)(v13 + 8 * ((v14 - v19) >> 1));
        v28 = v27 - v10;
        if ( v10 <= (char *)v22 )
        {
          if ( v10 != v27 )
            memmove((char *)v22 + v11 + 8 - v28, v10, v28);
        }
        else if ( v10 != v27 )
        {
          memmove(v22, v10, v28);
        }
      }
      else
      {
        v20 = 1;
        if ( v14 )
          v20 = a1[1];
        v21 = v14 + v20 + 2;
        if ( v21 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v13, v10, v14);
        v29 = sub_22077B0(8 * v21);
        v22 = (_QWORD *)(v29 + 8 * ((v21 - v19) >> 1));
        v23 = (const void *)a1[5];
        v24 = a1[9] + 8;
        if ( (const void *)v24 != v23 )
          memmove(v22, v23, v24 - (_QWORD)v23);
        j_j___libc_free_0(*a1, 8 * a1[1]);
        a1[1] = v21;
        *a1 = v29;
      }
      a1[5] = (__int64)v22;
      v25 = *v22;
      v9 = (char *)v22 + v11;
      a1[9] = (__int64)v22 + v11;
      a1[3] = v25;
      a1[4] = v25 + 480;
      v26 = *(_QWORD *)((char *)v22 + v11);
      a1[7] = v26;
      a1[8] = v26 + 480;
    }
    *((_QWORD *)v9 + 1) = sub_22077B0(480);
    v15 = a1[6];
    if ( v15 )
    {
      v16 = *a2;
      *(_DWORD *)(v15 + 8) = -1;
      *(_QWORD *)(v15 + 16) = -1;
      *(_QWORD *)v15 = v16;
      *(_QWORD *)(v15 + 32) = v15 + 48;
      *(_QWORD *)(v15 + 24) = 0;
      *(_QWORD *)(v15 + 40) = 0x100000000LL;
      *(_QWORD *)(v15 + 64) = v15 + 80;
      *(_QWORD *)(v15 + 72) = 0x100000000LL;
    }
    v7 = (__int64 *)(a1[9] + 8);
    a1[9] = (__int64)v7;
    v17 = *v7;
    v18 = *v7 + 480;
    a1[7] = *v7;
    a1[8] = v18;
    a1[6] = v17;
    goto LABEL_11;
  }
  if ( v4 )
  {
    v5 = *a2;
    *(_DWORD *)(v4 + 8) = -1;
    *(_QWORD *)(v4 + 16) = -1;
    *(_QWORD *)v4 = v5;
    *(_QWORD *)(v4 + 32) = v4 + 48;
    *(_QWORD *)(v4 + 24) = 0;
    *(_QWORD *)(v4 + 40) = 0x100000000LL;
    *(_QWORD *)(v4 + 64) = v4 + 80;
    *(_QWORD *)(v4 + 72) = 0x100000000LL;
    v4 = a1[6];
  }
  v6 = v4 + 96;
  v7 = (__int64 *)a1[9];
  a1[6] = v6;
  if ( v6 == a1[7] )
LABEL_11:
    v6 = *(v7 - 1) + 480;
  return v6 - 96;
}
