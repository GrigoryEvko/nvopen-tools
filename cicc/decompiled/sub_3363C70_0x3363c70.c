// Function: sub_3363C70
// Address: 0x3363c70
//
__int64 (*__fastcall sub_3363C70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6))(void)
{
  _QWORD *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 (*v13)(); // r8
  __int64 v14; // r9
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  const void *v19; // r13
  __int64 v20; // r15
  __int64 v21; // r12
  __int64 v22; // rax
  char *v23; // rbx
  signed __int64 v24; // rdx
  _QWORD *v25; // rdi
  bool (__fastcall *v26)(__int64); // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // r15d
  unsigned int v30; // r13d
  __int64 v31; // r12
  __int64 v32; // rdx
  __int64 (*v33)(); // rdi
  __int64 v34; // rax
  bool (__fastcall *v35)(__int64); // rdx
  _QWORD *v36; // rdi
  __int64 (*result)(void); // rax
  char v38; // bl
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // r15
  _BYTE *v42; // r12
  _BYTE *v43; // rdx
  __int64 v44; // rdi
  _QWORD **v45; // r13
  _BYTE *v46; // rsi
  __int64 v47; // rdi
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rdi
  void (*v53)(); // rdx
  int v54; // eax
  __int64 v55; // rax
  void (*v56)(void); // rax
  void (*v57)(void); // rdx
  void (*v58)(void); // rax
  __int64 v59; // [rsp+0h] [rbp-80h]
  unsigned int v60; // [rsp+14h] [rbp-6Ch]
  __int64 v61; // [rsp+28h] [rbp-58h] BYREF
  _BYTE *v62; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v63; // [rsp+38h] [rbp-48h]
  _BYTE *v64; // [rsp+40h] [rbp-40h]

  sub_3360940((unsigned __int64 *)a1, a2, a3, a4, a5, a6);
  (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 632) + 32LL))(*(_QWORD *)(a1 + 632), a1 + 48);
  v7 = (_QWORD *)(a1 + 72);
  sub_3363B70(a1, a1 + 72, v8, v9, v10, v11);
  v15 = *(_QWORD *)(a1 + 48);
  v16 = *(_QWORD *)(a1 + 56);
  if ( v15 == v16 )
  {
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v59 = a1 + 608;
    goto LABEL_11;
  }
  do
  {
    if ( !*(_DWORD *)(v15 + 48) )
    {
      v7 = (_QWORD *)v15;
      (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 632) + 88LL))(*(_QWORD *)(a1 + 632), v15);
      *(_BYTE *)(v15 + 249) |= 2u;
    }
    v15 += 256;
  }
  while ( v16 != v15 );
  v17 = *(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 48);
  v12 = a1 + 608;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v18 = v17 >> 8;
  v59 = a1 + 608;
  if ( v17 < 0 )
    sub_4262D8((__int64)"vector::reserve");
  v19 = *(const void **)(a1 + 608);
  if ( v18 <= (__int64)(*(_QWORD *)(a1 + 624) - (_QWORD)v19) >> 3 )
    goto LABEL_11;
  v20 = 8 * v18;
  v21 = *(_QWORD *)(a1 + 616) - (_QWORD)v19;
  if ( v18 )
  {
    v22 = sub_22077B0(8 * v18);
    v19 = *(const void **)(a1 + 608);
    v23 = (char *)v22;
    v24 = *(_QWORD *)(a1 + 616) - (_QWORD)v19;
    if ( v24 <= 0 )
      goto LABEL_9;
LABEL_77:
    memmove(v23, v19, v24);
    v7 = (_QWORD *)(*(_QWORD *)(a1 + 624) - (_QWORD)v19);
LABEL_78:
    j_j___libc_free_0((unsigned __int64)v19);
    goto LABEL_10;
  }
  v24 = *(_QWORD *)(a1 + 616) - (_QWORD)v19;
  v23 = 0;
  if ( v21 > 0 )
    goto LABEL_77;
LABEL_9:
  if ( v19 )
  {
    v7 = (_QWORD *)(*(_QWORD *)(a1 + 624) - (_QWORD)v19);
    goto LABEL_78;
  }
LABEL_10:
  *(_QWORD *)(a1 + 608) = v23;
  *(_QWORD *)(a1 + 616) = &v23[v21];
  *(_QWORD *)(a1 + 624) = &v23[v20];
LABEL_11:
  v60 = 0;
  while ( 1 )
  {
    v25 = *(_QWORD **)(a1 + 632);
    v26 = *(bool (__fastcall **)(__int64))(*v25 + 64LL);
    if ( v26 != sub_3363940 )
      break;
    if ( v25[7] == v25[6] )
      goto LABEL_27;
LABEL_14:
    v27 = *(_QWORD *)(a1 + 648);
    v28 = *(_QWORD *)(a1 + 640);
LABEL_15:
    v29 = (v27 - v28) >> 3;
    if ( v29 )
    {
      v30 = 0;
      while ( 1 )
      {
        v31 = *(_QWORD *)(v28 + 8LL * v30);
        if ( (*(_BYTE *)(v31 + 254) & 1) == 0 )
          sub_2F8F5D0(*(_QWORD *)(v28 + 8LL * v30), v7, v28, v12, (__int64)v13, v14);
        if ( *(_DWORD *)(v31 + 240) == v60 )
        {
          --v29;
          (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 632) + 88LL))(
            *(_QWORD *)(a1 + 632),
            *(_QWORD *)(*(_QWORD *)(a1 + 640) + 8LL * v30));
          v32 = *(_QWORD *)(*(_QWORD *)(a1 + 640) + 8LL * v30);
          *(_BYTE *)(v32 + 249) |= 2u;
          v7 = *(_QWORD **)(*(_QWORD *)(a1 + 648) - 8LL);
          *(_QWORD *)(*(_QWORD *)(a1 + 640) + 8LL * v30) = v7;
          *(_QWORD *)(a1 + 648) -= 8LL;
          if ( v29 == v30 )
            break;
        }
        else if ( v29 == ++v30 )
        {
          break;
        }
        v28 = *(_QWORD *)(a1 + 640);
      }
    }
    v33 = *(__int64 (**)())(a1 + 632);
    v34 = *(_QWORD *)v33;
    v13 = v33;
    v35 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v33 + 64LL);
    if ( v35 == sub_3363940 )
    {
      v12 = *((_QWORD *)v33 + 6);
      if ( *((_QWORD *)v33 + 7) == v12 )
        goto LABEL_25;
LABEL_33:
      v38 = 0;
      while ( 1 )
      {
        if ( v35 == sub_3363940 )
        {
          v12 = *((_QWORD *)v33 + 6);
          if ( *((_QWORD *)v33 + 7) == v12 )
            goto LABEL_59;
        }
        else
        {
          if ( ((unsigned __int8 (__fastcall *)(__int64 (*)(), _QWORD *, bool (__fastcall *)(__int64), __int64, __int64 (*)()))v35)(
                 v33,
                 v7,
                 v35,
                 v12,
                 v13) )
          {
LABEL_59:
            v42 = v63;
            v43 = v62;
            v41 = 0;
            if ( v63 == v62 )
              goto LABEL_60;
            goto LABEL_38;
          }
          v33 = *(__int64 (**)())(a1 + 632);
          v34 = *(_QWORD *)v33;
        }
        v39 = (*(__int64 (__fastcall **)(__int64 (*)(), _QWORD *, bool (__fastcall *)(__int64), __int64, __int64 (*)()))(v34 + 96))(
                v33,
                v7,
                v35,
                v12,
                v13);
        v40 = *(_QWORD *)(a1 + 664);
        v61 = v39;
        v41 = v39;
        v13 = *(__int64 (**)())(*(_QWORD *)v40 + 24LL);
        if ( v13 == sub_2EC0B50 )
          break;
        v7 = (_QWORD *)v39;
        v54 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v13)(v40, v39, 0);
        if ( !v54 )
        {
          v41 = v61;
          break;
        }
        v7 = v63;
        v38 |= v54 == 2;
        if ( v63 == v64 )
        {
          sub_2ECAD30((__int64)&v62, v63, &v61);
        }
        else
        {
          if ( v63 )
          {
            *(_QWORD *)v63 = v61;
            v7 = v63;
          }
          v63 = ++v7;
        }
        v33 = *(__int64 (**)())(a1 + 632);
        v34 = *(_QWORD *)v33;
        v35 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v33 + 64LL);
      }
      v42 = v63;
      v43 = v62;
      if ( v63 == v62 )
        goto LABEL_42;
LABEL_38:
      v44 = *(_QWORD *)(a1 + 632);
      v45 = (_QWORD **)v43;
      do
      {
        v7 = *v45++;
        (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v44 + 88LL))(v44, v7);
      }
      while ( v42 != (_BYTE *)v45 );
      v43 = v62;
      if ( v62 != v63 )
        v63 = v62;
LABEL_42:
      if ( v41 )
      {
        v61 = v41;
        v46 = *(_BYTE **)(a1 + 616);
        if ( v46 == *(_BYTE **)(a1 + 624) )
        {
          sub_2ECAD30(v59, v46, &v61);
          v47 = v61;
        }
        else
        {
          if ( v46 )
          {
            *(_QWORD *)v46 = v41;
            v46 = *(_BYTE **)(a1 + 616);
          }
          v47 = v41;
          *(_QWORD *)(a1 + 616) = v46 + 8;
        }
        sub_2F8F720(v47, (_QWORD *)v60, (__int64)v43, v12, (__int64)v13, v14);
        sub_3363B70(a1, v61, v48, v49, v50, v51);
        v7 = (_QWORD *)v61;
        *(_BYTE *)(v61 + 249) |= 4u;
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 632) + 120LL))(*(_QWORD *)(a1 + 632));
        v52 = *(_QWORD *)(a1 + 664);
        v53 = *(void (**)())(*(_QWORD *)v52 + 40LL);
        if ( v53 != nullsub_1619 )
        {
          v7 = (_QWORD *)v41;
          ((void (__fastcall *)(__int64, __int64))v53)(v52, v41);
        }
        v60 -= (*(_WORD *)(v41 + 252) == 0) - 1;
        continue;
      }
LABEL_60:
      v55 = **(_QWORD **)(a1 + 664);
      if ( v38 )
      {
        v57 = *(void (**)(void))(v55 + 96);
        if ( (char *)v57 == (char *)sub_2F39570 )
        {
          v58 = *(void (**)(void))(v55 + 80);
          if ( v58 != nullsub_1620 )
            v58();
        }
        else
        {
          v57();
        }
        v61 = 0;
        v7 = *(_QWORD **)(a1 + 616);
        if ( v7 == *(_QWORD **)(a1 + 624) )
        {
          sub_2F3A320(v59, v7, &v61);
        }
        else
        {
          if ( v7 )
          {
            *v7 = 0;
            v7 = *(_QWORD **)(a1 + 616);
          }
          *(_QWORD *)(a1 + 616) = ++v7;
        }
      }
      else
      {
        v56 = *(void (**)(void))(v55 + 80);
        if ( v56 != nullsub_1620 )
          v56();
      }
      ++v60;
    }
    else
    {
      if ( !((unsigned __int8 (__fastcall *)(__int64 (*)(), _QWORD *, bool (__fastcall *)(__int64), __int64, __int64 (*)()))v35)(
              v33,
              v7,
              v35,
              v12,
              v33) )
      {
        v33 = *(__int64 (**)())(a1 + 632);
        v34 = *(_QWORD *)v33;
        v35 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v33 + 64LL);
        goto LABEL_33;
      }
      v13 = *(__int64 (**)())(a1 + 632);
LABEL_25:
      v7 = 0;
      (*(void (__fastcall **)(__int64 (*)(), _QWORD))(*(_QWORD *)v13 + 120LL))(v13, 0);
      ++v60;
    }
  }
  if ( !((unsigned __int8 (*)(void))v26)() )
    goto LABEL_14;
LABEL_27:
  v27 = *(_QWORD *)(a1 + 648);
  v28 = *(_QWORD *)(a1 + 640);
  if ( v27 != v28 )
    goto LABEL_15;
  if ( v62 )
    j_j___libc_free_0((unsigned __int64)v62);
  v36 = *(_QWORD **)(a1 + 632);
  result = *(__int64 (**)(void))(*v36 + 56LL);
  if ( (char *)result != (char *)sub_3363930 )
    return (__int64 (*)(void))result();
  v36[2] = 0;
  return result;
}
