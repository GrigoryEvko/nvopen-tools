// Function: sub_37B5390
// Address: 0x37b5390
//
_QWORD *__fastcall sub_37B5390(__int64 a1)
{
  unsigned __int64 *v2; // r13
  unsigned __int64 *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // r15
  unsigned __int64 *v7; // r14
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // rax
  _QWORD *result; // rax
  char *v14; // r14
  char *v15; // rsi
  __int64 v16; // r13
  __int64 v17; // rcx
  __int64 v18; // r8
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // r14
  __int64 v26; // rax
  const void *v27; // rsi
  __int64 v28; // rcx
  __int64 *v29; // r15
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  char *v33; // r14
  size_t v34; // rdx
  __int64 v35; // [rsp+0h] [rbp-40h]
  __int64 v36; // [rsp+8h] [rbp-38h]
  __int64 v37; // [rsp+8h] [rbp-38h]
  __int64 v38; // [rsp+8h] [rbp-38h]

  v2 = *(unsigned __int64 **)(a1 + 192);
  v3 = &v2[6 * *(unsigned int *)(a1 + 200)];
  while ( v2 != v3 )
  {
    while ( 1 )
    {
      v3 -= 6;
      if ( (unsigned __int64 *)*v3 == v3 + 2 )
        break;
      _libc_free(*v3);
      if ( v2 == v3 )
        goto LABEL_5;
    }
  }
LABEL_5:
  v4 = *(_QWORD *)(a1 + 128);
  v5 = *(_QWORD *)(a1 + 152);
  *(_DWORD *)(a1 + 200) = 0;
  v6 = *(_QWORD *)(a1 + 144);
  v36 = v4;
  v7 = (unsigned __int64 *)(v5 + 8);
  v35 = *(_QWORD *)(a1 + 136);
  v8 = *(_QWORD *)(a1 + 184) + 8LL;
  if ( v8 > v5 + 8 )
  {
    do
    {
      v9 = *v7++;
      j_j___libc_free_0(v9);
    }
    while ( v8 > (unsigned __int64)v7 );
  }
  *(_QWORD *)(a1 + 184) = v5;
  *(_QWORD *)(a1 + 176) = v6;
  *(_QWORD *)(a1 + 160) = v36;
  *(_QWORD *)(a1 + 168) = v35;
  sub_37B4CB0(a1 + 16);
  v10 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 96) += 16LL;
  v11 = (_QWORD *)((v10 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( *(_QWORD *)(a1 + 24) >= (unsigned __int64)(v11 + 2) && v10 )
    *(_QWORD *)(a1 + 16) = v11 + 2;
  else
    v11 = (_QWORD *)sub_9D1E70(a1 + 16, 16, 16, 3);
  *v11 = 0;
  v11[1] = 0;
  v12 = *(_QWORD **)(a1 + 160);
  if ( v12 == (_QWORD *)(*(_QWORD *)(a1 + 176) - 8LL) )
  {
    v14 = *(char **)(a1 + 184);
    v15 = *(char **)(a1 + 152);
    v16 = v14 - v15;
    v17 = (v14 - v15) >> 3;
    if ( (((__int64)v12 - *(_QWORD *)(a1 + 168)) >> 3)
       + ((v17 - 1) << 6)
       + ((__int64)(*(_QWORD *)(a1 + 144) - *(_QWORD *)(a1 + 128)) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    v18 = *(_QWORD *)(a1 + 112);
    v19 = *(_QWORD *)(a1 + 120);
    if ( v19 - ((__int64)&v14[-v18] >> 3) <= 1 )
    {
      v23 = v17 + 2;
      if ( v19 > 2 * (v17 + 2) )
      {
        v33 = v14 + 8;
        v29 = (__int64 *)(v18 + 8 * ((v19 - v23) >> 1));
        v34 = v33 - v15;
        if ( v15 <= (char *)v29 )
        {
          if ( v15 != v33 )
            memmove((char *)v29 + v16 + 8 - v34, v15, v34);
        }
        else if ( v15 != v33 )
        {
          memmove(v29, v15, v34);
        }
      }
      else
      {
        v24 = 1;
        if ( v19 )
          v24 = *(_QWORD *)(a1 + 120);
        v25 = v19 + v24 + 2;
        if ( v25 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(0xFFFFFFFFFFFFFFFLL, v15, v19);
        v26 = sub_22077B0(8 * v25);
        v27 = *(const void **)(a1 + 152);
        v28 = v26;
        v29 = (__int64 *)(v26 + 8 * ((v25 - v23) >> 1));
        v30 = *(_QWORD *)(a1 + 184) + 8LL;
        if ( (const void *)v30 != v27 )
        {
          v37 = v28;
          memmove(v29, v27, v30 - (_QWORD)v27);
          v28 = v37;
        }
        v38 = v28;
        j_j___libc_free_0(*(_QWORD *)(a1 + 112));
        *(_QWORD *)(a1 + 120) = v25;
        *(_QWORD *)(a1 + 112) = v38;
      }
      *(_QWORD *)(a1 + 152) = v29;
      v31 = *v29;
      v14 = (char *)v29 + v16;
      *(_QWORD *)(a1 + 184) = (char *)v29 + v16;
      *(_QWORD *)(a1 + 136) = v31;
      *(_QWORD *)(a1 + 144) = v31 + 512;
      v32 = *(__int64 *)((char *)v29 + v16);
      *(_QWORD *)(a1 + 168) = v32;
      *(_QWORD *)(a1 + 176) = v32 + 512;
    }
    *((_QWORD *)v14 + 1) = sub_22077B0(0x200u);
    v20 = *(_QWORD **)(a1 + 160);
    if ( v20 )
      *v20 = v11;
    v21 = (_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL);
    *(_QWORD *)(a1 + 184) = v21;
    result = (_QWORD *)*v21;
    v22 = *v21 + 512LL;
    *(_QWORD *)(a1 + 168) = result;
    *(_QWORD *)(a1 + 176) = v22;
    *(_QWORD *)(a1 + 160) = result;
  }
  else
  {
    if ( v12 )
    {
      *v12 = v11;
      v12 = *(_QWORD **)(a1 + 160);
    }
    result = v12 + 1;
    *(_QWORD *)(a1 + 160) = result;
  }
  return result;
}
