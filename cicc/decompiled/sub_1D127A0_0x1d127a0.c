// Function: sub_1D127A0
// Address: 0x1d127a0
//
__int64 __fastcall sub_1D127A0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned __int64 v4; // rsi
  __int64 v5; // rbx
  unsigned __int64 v6; // rdi
  const void *v7; // r12
  char v8; // al
  __int64 v9; // rdx
  bool v10; // zf
  __int64 v11; // rax
  int v12; // r15d
  unsigned int v13; // r12d
  __int64 v14; // r13
  __int64 v15; // rdx
  char v16; // bl
  int v17; // eax
  __int64 *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // r13
  __int64 (*v22)(); // r8
  __int64 *v23; // rdx
  __int64 v24; // r12
  __int64 *v25; // r15
  __int64 v26; // rsi
  _BYTE *v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rdi
  void (*v30)(); // rdx
  __int64 v31; // rax
  void (*v32)(void); // rax
  void (*v33)(void); // rdx
  void (*v34)(void); // rax
  _BYTE *v35; // rsi
  __int64 v37; // rbx
  __int64 v38; // rax
  char *v39; // r13
  signed __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // [rsp+0h] [rbp-80h]
  unsigned int v43; // [rsp+14h] [rbp-6Ch]
  __int64 *v44; // [rsp+18h] [rbp-68h]
  __int64 v45; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v46; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v47; // [rsp+38h] [rbp-48h]
  __int64 *v48; // [rsp+40h] [rbp-40h]

  sub_1D10D90((__int64 *)a1, *(_QWORD *)(a1 + 704));
  (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 664) + 32LL))(*(_QWORD *)(a1 + 664), a1 + 48);
  sub_1D12510(a1, a1 + 72);
  v2 = *(_QWORD *)(a1 + 48);
  v3 = *(_QWORD *)(a1 + 56) - v2;
  v4 = 0xF0F0F0F0F0F0F0F1LL * (v3 >> 4);
  if ( (_DWORD)v4 )
  {
    v5 = 0;
    do
    {
      if ( !*(_DWORD *)(v2 + v5 + 40) )
      {
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 664) + 88LL))(*(_QWORD *)(a1 + 664));
        *(_BYTE *)(*(_QWORD *)(a1 + 48) + v5 + 229) |= 2u;
        v2 = *(_QWORD *)(a1 + 48);
      }
      v5 += 272;
    }
    while ( 272LL * (unsigned int)v4 != v5 );
    v3 = *(_QWORD *)(a1 + 56) - v2;
    v6 = 0xF0F0F0F0F0F0F0F1LL * (v3 >> 4);
  }
  else
  {
    v6 = 0xF0F0F0F0F0F0F0F1LL * (v3 >> 4);
  }
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v42 = a1 + 640;
  if ( v3 < 0 )
    sub_4262D8((__int64)"vector::reserve");
  v7 = *(const void **)(a1 + 640);
  if ( v6 <= (__int64)(*(_QWORD *)(a1 + 656) - (_QWORD)v7) >> 3 )
    goto LABEL_9;
  v37 = *(_QWORD *)(a1 + 648) - (_QWORD)v7;
  if ( v6 )
  {
    v38 = sub_22077B0(8 * v6);
    v7 = *(const void **)(a1 + 640);
    v39 = (char *)v38;
    v40 = *(_QWORD *)(a1 + 648) - (_QWORD)v7;
    if ( v40 <= 0 )
      goto LABEL_61;
LABEL_66:
    memmove(v39, v7, v40);
    v41 = *(_QWORD *)(a1 + 656) - (_QWORD)v7;
LABEL_67:
    j_j___libc_free_0(v7, v41);
    goto LABEL_62;
  }
  v40 = *(_QWORD *)(a1 + 648) - (_QWORD)v7;
  v39 = 0;
  if ( v37 > 0 )
    goto LABEL_66;
LABEL_61:
  if ( v7 )
  {
    v41 = *(_QWORD *)(a1 + 656) - (_QWORD)v7;
    goto LABEL_67;
  }
LABEL_62:
  *(_QWORD *)(a1 + 640) = v39;
  *(_QWORD *)(a1 + 648) = &v39[v37];
  *(_QWORD *)(a1 + 656) = &v39[8 * v6];
LABEL_9:
  v43 = 0;
  while ( 1 )
  {
    v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 664) + 64LL))(*(_QWORD *)(a1 + 664));
    v9 = *(_QWORD *)(a1 + 672);
    v10 = v8 == 0;
    v11 = *(_QWORD *)(a1 + 680);
    if ( !v10 && v11 == v9 )
      break;
    v12 = (v11 - v9) >> 3;
    if ( v12 )
    {
      v13 = 0;
      while ( 1 )
      {
        v14 = *(_QWORD *)(v9 + 8LL * v13);
        if ( (*(_BYTE *)(v14 + 236) & 1) == 0 )
          sub_1F01DD0(*(_QWORD *)(v9 + 8LL * v13));
        if ( *(_DWORD *)(v14 + 240) == v43 )
        {
          --v12;
          (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 664) + 88LL))(
            *(_QWORD *)(a1 + 664),
            *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8LL * v13));
          v15 = *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8LL * v13);
          *(_BYTE *)(v15 + 229) |= 2u;
          *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8LL * v13) = *(_QWORD *)(*(_QWORD *)(a1 + 680) - 8LL);
          *(_QWORD *)(a1 + 680) -= 8LL;
          if ( v12 == v13 )
            break;
        }
        else if ( v12 == ++v13 )
        {
          break;
        }
        v9 = *(_QWORD *)(a1 + 672);
      }
    }
    v16 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 664) + 64LL))(*(_QWORD *)(a1 + 664));
    if ( v16 )
    {
      (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 664) + 120LL))(*(_QWORD *)(a1 + 664), 0);
      ++v43;
    }
    else
    {
      while ( 1 )
      {
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 664) + 64LL))(*(_QWORD *)(a1 + 664)) )
        {
          v23 = v46;
          v21 = 0;
          v44 = v47;
          if ( v47 == v46 )
            goto LABEL_44;
          goto LABEL_30;
        }
        v19 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 664) + 96LL))(*(_QWORD *)(a1 + 664));
        v20 = *(_QWORD *)(a1 + 696);
        v45 = v19;
        v21 = v19;
        v22 = *(__int64 (**)())(*(_QWORD *)v20 + 24LL);
        if ( v22 == sub_1D00B90 )
          break;
        v17 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v22)(v20, v19, 0);
        if ( !v17 )
        {
          v21 = v45;
          break;
        }
        v18 = v47;
        v16 |= v17 == 2;
        if ( v47 == v48 )
        {
          sub_1CFD630((__int64)&v46, v47, &v45);
        }
        else
        {
          if ( v47 )
          {
            *v47 = v45;
            v18 = v47;
          }
          v47 = v18 + 1;
        }
      }
      v23 = v46;
      v44 = v47;
      if ( v47 == v46 )
        goto LABEL_34;
LABEL_30:
      v24 = *(_QWORD *)(a1 + 664);
      v25 = v23;
      do
      {
        v26 = *v25++;
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v24 + 88LL))(v24, v26);
      }
      while ( v44 != v25 );
      if ( v46 != v47 )
        v47 = v46;
LABEL_34:
      if ( v21 )
      {
        v45 = v21;
        v27 = *(_BYTE **)(a1 + 648);
        if ( v27 == *(_BYTE **)(a1 + 656) )
        {
          sub_1CFD630(v42, v27, &v45);
          v28 = v45;
        }
        else
        {
          if ( v27 )
          {
            *(_QWORD *)v27 = v21;
            v27 = *(_BYTE **)(a1 + 648);
          }
          v28 = v21;
          *(_QWORD *)(a1 + 648) = v27 + 8;
        }
        sub_1F01F20(v28, v43);
        sub_1D12510(a1, v45);
        *(_BYTE *)(v45 + 229) |= 4u;
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 664) + 120LL))(*(_QWORD *)(a1 + 664));
        v29 = *(_QWORD *)(a1 + 696);
        v30 = *(void (**)())(*(_QWORD *)v29 + 40LL);
        if ( v30 != nullsub_679 )
          ((void (__fastcall *)(__int64, __int64))v30)(v29, v21);
        v43 -= (*(_WORD *)(v21 + 226) == 0) - 1;
        continue;
      }
LABEL_44:
      v31 = **(_QWORD **)(a1 + 696);
      if ( v16 )
      {
        v33 = *(void (**)(void))(v31 + 96);
        if ( (char *)v33 == (char *)sub_1D123B0 )
        {
          v34 = *(void (**)(void))(v31 + 80);
          if ( v34 != nullsub_683 )
            v34();
        }
        else
        {
          v33();
        }
        v45 = 0;
        v35 = *(_BYTE **)(a1 + 648);
        if ( v35 == *(_BYTE **)(a1 + 656) )
        {
          sub_1D12610(v42, v35, &v45);
        }
        else
        {
          if ( v35 )
          {
            *(_QWORD *)v35 = 0;
            v35 = *(_BYTE **)(a1 + 648);
          }
          *(_QWORD *)(a1 + 648) = v35 + 8;
        }
      }
      else
      {
        v32 = *(void (**)(void))(v31 + 80);
        if ( v32 != nullsub_683 )
          v32();
      }
      ++v43;
    }
  }
  if ( v46 )
    j_j___libc_free_0(v46, (char *)v48 - (char *)v46);
  return (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 664) + 56LL))(*(_QWORD *)(a1 + 664));
}
