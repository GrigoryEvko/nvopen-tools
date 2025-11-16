// Function: sub_F52DE0
// Address: 0xf52de0
//
__int64 __fastcall sub_F52DE0(unsigned __int8 *a1, unsigned __int8 *a2, __int64 a3, char a4, int a5)
{
  unsigned __int64 *v6; // rbx
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  unsigned int v12; // r13d
  unsigned __int64 v13; // r12
  __int64 *v14; // rbx
  __int64 v15; // r13
  __int64 v16; // r14
  _QWORD *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned __int8 *v20; // rsi
  _QWORD *v21; // rax
  _QWORD *v22; // r12
  __int64 v24; // r12
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 *v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rcx
  unsigned __int64 *v34; // [rsp+18h] [rbp-68h]
  __int64 *v35; // [rsp+20h] [rbp-60h]
  unsigned __int64 v37; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int64 v38; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v39[7]; // [rsp+48h] [rbp-38h] BYREF

  v6 = &v37;
  sub_AE74C0(&v37, (__int64)a1);
  v7 = (__int64)a1;
  sub_AE7690(&v38, (__int64)a1);
  if ( (v37 & 4) != 0 )
  {
    v6 = *(unsigned __int64 **)(v37 & 0xFFFFFFFFFFFFFFF8LL);
    v34 = &v6[*(unsigned int *)((v37 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
    if ( v6 == v34 )
      goto LABEL_3;
  }
  else
  {
    if ( (v37 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_3;
    v34 = &v38;
  }
  do
  {
    v24 = *v6;
    v25 = sub_B0DAC0(*(_QWORD **)(*(_QWORD *)(*v6 + 32 * (2LL - (*(_DWORD *)(*v6 + 4) & 0x7FFFFFF))) + 24LL), a4, a5);
    v26 = *(_QWORD *)(v25 + 8);
    v27 = (__int64 *)(v26 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v26 & 4) != 0 )
      v27 = (__int64 *)*v27;
    v28 = sub_B9F6F0(v27, (_BYTE *)v25);
    v29 = v24 + 32 * (2LL - (*(_DWORD *)(v24 + 4) & 0x7FFFFFF));
    if ( *(_QWORD *)v29 )
    {
      v30 = *(_QWORD *)(v29 + 8);
      **(_QWORD **)(v29 + 16) = v30;
      if ( v30 )
        *(_QWORD *)(v30 + 16) = *(_QWORD *)(v29 + 16);
    }
    *(_QWORD *)v29 = v28;
    if ( v28 )
    {
      v31 = *(_QWORD *)(v28 + 16);
      *(_QWORD *)(v29 + 8) = v31;
      if ( v31 )
        *(_QWORD *)(v31 + 16) = v29 + 8;
      *(_QWORD *)(v29 + 16) = v28 + 16;
      *(_QWORD *)(v28 + 16) = v29;
    }
    v7 = (__int64)a1;
    ++v6;
    sub_B59720(v24, (__int64)a1, a2);
  }
  while ( v34 != v6 );
LABEL_3:
  v8 = v38;
  v9 = v38 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v38 & 4) == 0 )
  {
    if ( !v9 )
      goto LABEL_5;
    v14 = (__int64 *)&v38;
    v35 = v39;
    do
    {
LABEL_13:
      v15 = *v14;
      v16 = *v14 + 80;
      v17 = (_QWORD *)sub_B11F60(v16);
      v18 = sub_B0DAC0(v17, a4, a5);
      sub_B11F20(v39, v18);
      v19 = *(_QWORD *)(v15 + 80);
      if ( v19 )
        sub_B91220(v16, v19);
      v20 = (unsigned __int8 *)v39[0];
      *(_QWORD *)(v15 + 80) = v39[0];
      if ( v20 )
        sub_B976B0((__int64)v39, v20, v16);
      v7 = (__int64)a1;
      ++v14;
      sub_B13360(v15, a1, a2, 0);
    }
    while ( v35 != v14 );
    v10 = v37;
    v8 = v38;
    v11 = v37 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v37 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_19;
    goto LABEL_6;
  }
  v14 = *(__int64 **)v9;
  v35 = (__int64 *)(*(_QWORD *)v9 + 8LL * *(unsigned int *)(v9 + 8));
  if ( *(__int64 **)v9 != v35 )
    goto LABEL_13;
LABEL_5:
  v10 = v37;
  v11 = v37 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v37 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
LABEL_19:
    v13 = v8 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v12 = 1;
      if ( (v8 & 4) == 0 )
        goto LABEL_25;
      LOBYTE(v12) = *(_DWORD *)(v13 + 8) != 0;
LABEL_22:
      if ( *(_QWORD *)v13 != v13 + 16 )
        _libc_free(*(_QWORD *)v13, v7);
      v7 = 48;
      j_j___libc_free_0(v13, 48);
      v10 = v37;
LABEL_25:
      if ( !v10 )
        return v12;
      goto LABEL_26;
    }
    v12 = 0;
    if ( !v8 )
      goto LABEL_25;
LABEL_9:
    if ( (v8 & 4) == 0 )
      goto LABEL_25;
    v13 = v8 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_25;
    goto LABEL_22;
  }
LABEL_6:
  if ( (v10 & 4) != 0 && !*(_DWORD *)(v11 + 8) )
    goto LABEL_19;
  v12 = 1;
  if ( v8 )
    goto LABEL_9;
LABEL_26:
  if ( (v10 & 4) != 0 )
  {
    v21 = (_QWORD *)(v10 & 0xFFFFFFFFFFFFFFF8LL);
    v22 = v21;
    if ( v21 )
    {
      if ( (_QWORD *)*v21 != v21 + 2 )
        _libc_free(*v21, v7);
      j_j___libc_free_0(v22, 48);
    }
  }
  return v12;
}
