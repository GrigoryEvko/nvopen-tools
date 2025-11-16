// Function: sub_389B1F0
// Address: 0x389b1f0
//
__int64 *__fastcall sub_389B1F0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 *v5; // r13
  __int64 v6; // rcx
  __int64 *v7; // rsi
  unsigned __int64 *v8; // rax
  unsigned __int64 *v9; // rdx
  unsigned __int64 v10; // r8
  __int64 v11; // rax
  _BYTE *v12; // r8
  __int64 v13; // r14
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  bool v18; // al
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rbx
  int *v23; // rdi
  int *v24; // rax
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // r8
  __int64 v27; // rsi
  __int64 v28; // r9
  __int64 v29; // rdi
  __int64 v30; // rsi
  int *v31; // rdi
  int *v32; // rax
  int *v33; // [rsp+0h] [rbp-50h]
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+8h] [rbp-48h]
  __int64 *v36; // [rsp+18h] [rbp-38h] BYREF

  if ( *(_QWORD *)(a2 + 8) )
    v5 = sub_389AAD0((_QWORD *)a1, a2, a3);
  else
    v5 = sub_389B190((__int64 *)a1, (__int64)(*(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112)) >> 3, a3);
  if ( !v5 )
    return v5;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = v5 + 3;
  v8 = (unsigned __int64 *)v5[4];
  v9 = (unsigned __int64 *)(v6 + 72);
  if ( (__int64 *)(v6 + 72) != v5 + 3 && v9 != v8 && v7 != (__int64 *)v8 )
  {
    v10 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((v5[3] & 0xFFFFFFFFFFFFFFF8LL) + 8) = v8;
    *v8 = *v8 & 7 | v5[3] & 0xFFFFFFFFFFFFFFF8LL;
    v11 = *(_QWORD *)(v6 + 72);
    *(_QWORD *)(v10 + 8) = v9;
    v11 &= 0xFFFFFFFFFFFFFFF8LL;
    v5[3] = v11 | v5[3] & 7;
    *(_QWORD *)(v11 + 8) = v7;
    *(_QWORD *)(v6 + 72) = v10 | *(_QWORD *)(v6 + 72) & 7LL;
  }
  if ( !*(_QWORD *)(a2 + 8) )
  {
    v12 = *(_BYTE **)(a1 + 120);
    v13 = a1 + 72;
    v14 = a1 + 72;
    v15 = (__int64)&v12[-*(_QWORD *)(a1 + 112)] >> 3;
    if ( *(_QWORD *)(a1 + 80) )
    {
      v16 = *(_QWORD *)(a1 + 80);
      while ( 1 )
      {
        while ( (unsigned int)v15 > *(_DWORD *)(v16 + 32) )
        {
          v16 = *(_QWORD *)(v16 + 24);
          if ( !v16 )
            goto LABEL_15;
        }
        v17 = *(_QWORD *)(v16 + 16);
        if ( (unsigned int)v15 >= *(_DWORD *)(v16 + 32) )
          break;
        v14 = v16;
        v16 = *(_QWORD *)(v16 + 16);
        if ( !v17 )
        {
LABEL_15:
          v18 = v13 == v14;
          goto LABEL_16;
        }
      }
      v27 = *(_QWORD *)(v16 + 24);
      if ( v27 )
      {
        do
        {
          while ( 1 )
          {
            v28 = *(_QWORD *)(v27 + 16);
            v29 = *(_QWORD *)(v27 + 24);
            if ( (unsigned int)v15 < *(_DWORD *)(v27 + 32) )
              break;
            v27 = *(_QWORD *)(v27 + 24);
            if ( !v29 )
              goto LABEL_36;
          }
          v14 = v27;
          v27 = *(_QWORD *)(v27 + 16);
        }
        while ( v28 );
      }
LABEL_36:
      while ( v17 )
      {
        while ( 1 )
        {
          v30 = *(_QWORD *)(v17 + 24);
          if ( (unsigned int)v15 <= *(_DWORD *)(v17 + 32) )
            break;
          v17 = *(_QWORD *)(v17 + 24);
          if ( !v30 )
            goto LABEL_39;
        }
        v16 = v17;
        v17 = *(_QWORD *)(v17 + 16);
      }
LABEL_39:
      if ( *(_QWORD *)(a1 + 88) != v16 || v13 != v14 )
      {
        if ( v14 != v16 )
        {
          do
          {
            v35 = v14;
            v31 = (int *)v16;
            v16 = sub_220EF30(v16);
            v32 = sub_220F330(v31, (_QWORD *)(a1 + 72));
            j_j___libc_free_0((unsigned __int64)v32);
            v14 = v35;
            --*(_QWORD *)(a1 + 104);
          }
          while ( v16 != v35 );
          v12 = *(_BYTE **)(a1 + 120);
        }
        goto LABEL_19;
      }
    }
    else
    {
      v18 = 1;
LABEL_16:
      if ( *(_QWORD *)(a1 + 88) != v14 || !v18 )
        goto LABEL_19;
    }
    sub_3887900(*(_QWORD *)(a1 + 80));
    *(_QWORD *)(a1 + 88) = v13;
    v12 = *(_BYTE **)(a1 + 120);
    *(_QWORD *)(a1 + 80) = 0;
    *(_QWORD *)(a1 + 96) = v13;
    *(_QWORD *)(a1 + 104) = 0;
LABEL_19:
    v36 = v5;
    if ( *(_BYTE **)(a1 + 128) == v12 )
    {
      sub_12879C0(a1 + 112, v12, &v36);
    }
    else
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = v5;
        v12 = *(_BYTE **)(a1 + 120);
      }
      *(_QWORD *)(a1 + 120) = v12 + 8;
    }
    return v5;
  }
  v20 = sub_3894C30(a1 + 16, a2);
  v34 = v21;
  v22 = v20;
  if ( v20 == *(_QWORD *)(a1 + 40) && v21 == a1 + 24 )
  {
    sub_38882E0(*(_QWORD **)(a1 + 32));
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 40) = v34;
    *(_QWORD *)(a1 + 48) = v34;
  }
  else if ( v21 != v20 )
  {
    do
    {
      v23 = (int *)v22;
      v22 = sub_220EF30(v22);
      v24 = sub_220F330(v23, (_QWORD *)(a1 + 24));
      v25 = *((_QWORD *)v24 + 4);
      v26 = (unsigned __int64)v24;
      if ( (int *)v25 != v24 + 12 )
      {
        v33 = v24;
        j_j___libc_free_0(v25);
        v26 = (unsigned __int64)v33;
      }
      j_j___libc_free_0(v26);
      --*(_QWORD *)(a1 + 56);
    }
    while ( v34 != v22 );
  }
  return v5;
}
