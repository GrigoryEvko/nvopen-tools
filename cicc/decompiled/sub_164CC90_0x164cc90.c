// Function: sub_164CC90
// Address: 0x164cc90
//
__int64 __fastcall sub_164CC90(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 v8; // r9
  unsigned int v9; // r8d
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 *v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  void (*v16)(); // rax
  __int64 result; // rax
  int v18; // r11d
  __int64 *v19; // rdx
  int v20; // eax
  int v21; // ecx
  int v22; // eax
  int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // eax
  __int64 v26; // rdi
  int v27; // r10d
  __int64 *v28; // r9
  int v29; // eax
  int v30; // eax
  __int64 v31; // rdi
  __int64 *v32; // r8
  unsigned int v33; // r14d
  int v34; // r9d
  __int64 v35; // rsi
  __int64 *v36; // r15
  __int64 v37; // [rsp+0h] [rbp-50h] BYREF
  __int64 *v38; // [rsp+8h] [rbp-48h]
  __int64 v39; // [rsp+10h] [rbp-40h]

  v4 = sub_16498A0(a1);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(*(_QWORD *)v4 + 2664LL);
  v7 = *(_QWORD *)v4 + 2640LL;
  if ( v6 )
  {
    v8 = *(_QWORD *)(v5 + 2648);
    v9 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a1 )
    {
      v12 = (__int64 *)v10[1];
      goto LABEL_4;
    }
    v18 = 1;
    v19 = 0;
    while ( v11 != -8 )
    {
      if ( v11 != -16 || v19 )
        v10 = v19;
      v9 = (v6 - 1) & (v18 + v9);
      v36 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v36;
      if ( *v36 == a1 )
      {
        v12 = (__int64 *)v36[1];
        goto LABEL_4;
      }
      ++v18;
      v19 = v10;
      v10 = (__int64 *)(v8 + 16LL * v9);
    }
    if ( !v19 )
      v19 = v10;
    v20 = *(_DWORD *)(v5 + 2656);
    ++*(_QWORD *)(v5 + 2640);
    v21 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(v5 + 2660) - v21 > v6 >> 3 )
        goto LABEL_29;
      sub_164B930(v7, v6);
      v29 = *(_DWORD *)(v5 + 2664);
      if ( v29 )
      {
        v30 = v29 - 1;
        v31 = *(_QWORD *)(v5 + 2648);
        v32 = 0;
        v33 = v30 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v34 = 1;
        v21 = *(_DWORD *)(v5 + 2656) + 1;
        v19 = (__int64 *)(v31 + 16LL * v33);
        v35 = *v19;
        if ( *v19 != a1 )
        {
          while ( v35 != -8 )
          {
            if ( v35 == -16 && !v32 )
              v32 = v19;
            v33 = v30 & (v34 + v33);
            v19 = (__int64 *)(v31 + 16LL * v33);
            v35 = *v19;
            if ( *v19 == a1 )
              goto LABEL_29;
            ++v34;
          }
          if ( v32 )
            v19 = v32;
        }
        goto LABEL_29;
      }
LABEL_62:
      ++*(_DWORD *)(v5 + 2656);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(v5 + 2640);
  }
  sub_164B930(v7, 2 * v6);
  v22 = *(_DWORD *)(v5 + 2664);
  if ( !v22 )
    goto LABEL_62;
  v23 = v22 - 1;
  v24 = *(_QWORD *)(v5 + 2648);
  v25 = (v22 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v21 = *(_DWORD *)(v5 + 2656) + 1;
  v19 = (__int64 *)(v24 + 16LL * v25);
  v26 = *v19;
  if ( *v19 != a1 )
  {
    v27 = 1;
    v28 = 0;
    while ( v26 != -8 )
    {
      if ( !v28 && v26 == -16 )
        v28 = v19;
      v25 = v23 & (v27 + v25);
      v19 = (__int64 *)(v24 + 16LL * v25);
      v26 = *v19;
      if ( *v19 == a1 )
        goto LABEL_29;
      ++v27;
    }
    if ( v28 )
      v19 = v28;
  }
LABEL_29:
  *(_DWORD *)(v5 + 2656) = v21;
  if ( *v19 != -8 )
    --*(_DWORD *)(v5 + 2660);
  *v19 = a1;
  v12 = 0;
  v19[1] = 0;
LABEL_4:
  v13 = v12[2];
  v37 = 0;
  v38 = 0;
  v39 = v13;
  if ( v13 != 0 && v13 != -8 && v13 != -16 )
    sub_1649AC0((unsigned __int64 *)&v37, *v12 & 0xFFFFFFFFFFFFFFF8LL);
  do
  {
    while ( 1 )
    {
      sub_1649B30(&v37);
      sub_1649AF0(&v37, (__int64)v12);
      v15 = (*v12 >> 1) & 3;
      if ( v15 != 1 )
      {
        if ( (_DWORD)v15 == 3 )
        {
          v14 = v12[2];
          if ( a2 != v14 )
          {
            if ( v14 != -8 && v14 != 0 && v14 != -16 )
              sub_1649B30(v12);
            v12[2] = a2;
            if ( a2 != 0 && a2 != -8 && a2 != -16 )
              sub_164C220((__int64)v12);
          }
        }
        goto LABEL_15;
      }
      v16 = *(void (**)())(*(v12 - 1) + 16);
      if ( v16 != nullsub_516 )
        break;
LABEL_15:
      v12 = v38;
      if ( !v38 )
        goto LABEL_19;
    }
    ((void (__fastcall *)(__int64 *, __int64))v16)(v12 - 1, a2);
    v12 = v38;
  }
  while ( v38 );
LABEL_19:
  result = v39;
  if ( v39 != 0 && v39 != -8 && v39 != -16 )
    return sub_1649B30(&v37);
  return result;
}
