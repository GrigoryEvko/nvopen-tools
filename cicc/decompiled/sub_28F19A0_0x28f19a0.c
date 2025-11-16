// Function: sub_28F19A0
// Address: 0x28f19a0
//
__int64 __fastcall sub_28F19A0(__int64 a1, _QWORD *a2)
{
  unsigned int v4; // esi
  _QWORD *v5; // r14
  int v6; // edx
  bool v7; // zf
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int64 *v10; // rdi
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // r8
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  char *v16; // r14
  char *v17; // rsi
  __int64 v18; // r13
  __int64 v19; // rcx
  __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  unsigned __int64 *v22; // rdi
  __int64 *v23; // rdx
  __int64 v24; // rdx
  int v25; // r10d
  int v26; // eax
  __int64 v27; // r15
  char *v28; // r14
  __int64 *v29; // r15
  size_t v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // r14
  __int64 v35; // rax
  const void *v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // [rsp+8h] [rbp-68h]
  _QWORD *v39; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v40[10]; // [rsp+20h] [rbp-50h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v12 = a2[2];
    v13 = *(_QWORD *)(a1 + 8);
    result = (v4 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v14 = (_QWORD *)(v13 + 24 * result);
    v15 = v14[2];
    if ( v12 == v15 )
      return result;
    v25 = 1;
    v5 = 0;
    while ( v15 != -4096 )
    {
      if ( v15 != -8192 || v5 )
        v14 = v5;
      result = (v4 - 1) & (v25 + (_DWORD)result);
      v15 = *(_QWORD *)(v13 + 24LL * (unsigned int)result + 16);
      if ( v12 == v15 )
        return result;
      v5 = v14;
      ++v25;
      v14 = (_QWORD *)(v13 + 24LL * (unsigned int)result);
    }
    v26 = *(_DWORD *)(a1 + 16);
    if ( !v5 )
      v5 = v14;
    ++*(_QWORD *)a1;
    v6 = v26 + 1;
    v39 = v5;
    if ( 4 * (v26 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 20) - v6 > v4 >> 3 )
        goto LABEL_5;
      goto LABEL_4;
    }
  }
  else
  {
    ++*(_QWORD *)a1;
    v39 = 0;
  }
  v4 *= 2;
LABEL_4:
  sub_28F1640(a1, v4);
  sub_28EF080(a1, (__int64)a2, &v39);
  v5 = v39;
  v6 = *(_DWORD *)(a1 + 16) + 1;
LABEL_5:
  *(_DWORD *)(a1 + 16) = v6;
  v40[2] = -4096;
  v7 = v5[2] == -4096;
  v40[0] = 0;
  v40[1] = 0;
  if ( !v7 )
    --*(_DWORD *)(a1 + 20);
  sub_D68D70(v40);
  v8 = a2[2];
  v9 = v5[2];
  if ( v8 != v9 )
  {
    if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
      sub_BD60C0(v5);
    v5[2] = v8;
    if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
      sub_BD73F0((__int64)v5);
  }
  v10 = *(unsigned __int64 **)(a1 + 80);
  result = *(_QWORD *)(a1 + 96) - 24LL;
  if ( v10 == (unsigned __int64 *)result )
  {
    v16 = *(char **)(a1 + 104);
    v17 = *(char **)(a1 + 72);
    v18 = v16 - v17;
    v19 = (v16 - v17) >> 3;
    if ( 21 * (v19 - 1)
       - 0x5555555555555555LL * (((__int64)v10 - *(_QWORD *)(a1 + 88)) >> 3)
       - 0x5555555555555555LL * ((__int64)(*(_QWORD *)(a1 + 64) - *(_QWORD *)(a1 + 48)) >> 3) == 0x555555555555555LL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    v20 = *(_QWORD *)(a1 + 32);
    v21 = *(_QWORD *)(a1 + 40);
    if ( v21 - ((__int64)&v16[-v20] >> 3) <= 1 )
    {
      v27 = v19 + 2;
      if ( v21 <= 2 * (v19 + 2) )
      {
        v33 = 1;
        if ( v21 )
          v33 = *(_QWORD *)(a1 + 40);
        v34 = v21 + v33 + 2;
        if ( v34 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v20, v17, v21);
        v35 = sub_22077B0(8 * v34);
        v36 = *(const void **)(a1 + 72);
        v38 = v35;
        v29 = (__int64 *)(v35 + 8 * ((v34 - v27) >> 1));
        v37 = *(_QWORD *)(a1 + 104) + 8LL;
        if ( (const void *)v37 != v36 )
          memmove(v29, v36, v37 - (_QWORD)v36);
        j_j___libc_free_0(*(_QWORD *)(a1 + 32));
        *(_QWORD *)(a1 + 40) = v34;
        *(_QWORD *)(a1 + 32) = v38;
      }
      else
      {
        v28 = v16 + 8;
        v29 = (__int64 *)(v20 + 8 * ((v21 - v27) >> 1));
        v30 = v28 - v17;
        if ( v17 <= (char *)v29 )
        {
          if ( v28 != v17 )
            memmove((char *)v29 + v18 - v30 + 8, v17, v30);
        }
        else if ( v28 != v17 )
        {
          memmove(v29, v17, v30);
        }
      }
      *(_QWORD *)(a1 + 72) = v29;
      v31 = *v29;
      v16 = (char *)v29 + v18;
      *(_QWORD *)(a1 + 104) = (char *)v29 + v18;
      *(_QWORD *)(a1 + 56) = v31;
      *(_QWORD *)(a1 + 64) = v31 + 504;
      v32 = *(__int64 *)((char *)v29 + v18);
      *(_QWORD *)(a1 + 88) = v32;
      *(_QWORD *)(a1 + 96) = v32 + 504;
    }
    *((_QWORD *)v16 + 1) = sub_22077B0(0x1F8u);
    v22 = *(unsigned __int64 **)(a1 + 80);
    if ( v22 )
      sub_D68CD0(v22, 0, a2);
    v23 = (__int64 *)(*(_QWORD *)(a1 + 104) + 8LL);
    *(_QWORD *)(a1 + 104) = v23;
    result = *v23;
    v24 = *v23 + 504;
    *(_QWORD *)(a1 + 88) = result;
    *(_QWORD *)(a1 + 96) = v24;
    *(_QWORD *)(a1 + 80) = result;
  }
  else
  {
    if ( v10 )
    {
      *v10 = 0;
      result = a2[2];
      v10[1] = 0;
      v10[2] = result;
      if ( result != 0 && result != -4096 && result != -8192 )
        result = sub_BD6050(v10, *a2 & 0xFFFFFFFFFFFFFFF8LL);
      v10 = *(unsigned __int64 **)(a1 + 80);
    }
    *(_QWORD *)(a1 + 80) = v10 + 3;
  }
  return result;
}
