// Function: sub_1673870
// Address: 0x1673870
//
__int64 __fastcall sub_1673870(__int64 a1, __int64 a2)
{
  unsigned int v4; // ebx
  __int64 v5; // r8
  int v6; // ebx
  __int64 v7; // r14
  int v8; // ebx
  __int64 *v9; // r13
  int v10; // eax
  __int64 v11; // rax
  unsigned int v12; // ebx
  __int64 *v13; // r14
  __int64 result; // rax
  int v15; // esi
  __int64 v16; // r9
  __int64 *v17; // rdi
  __int64 v18; // r8
  int v19; // edi
  int v20; // r10d
  int v21; // eax
  int v22; // ebx
  __int64 v23; // r14
  int v24; // ebx
  bool v25; // al
  bool v26; // al
  bool v27; // al
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  int v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  int v34; // [rsp+18h] [rbp-48h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  int v36; // [rsp+18h] [rbp-48h]
  __int64 *v37; // [rsp+20h] [rbp-40h]
  __int64 v38; // [rsp+20h] [rbp-40h]
  unsigned int v39; // [rsp+2Ch] [rbp-34h]
  unsigned int v40; // [rsp+2Ch] [rbp-34h]
  unsigned int v41; // [rsp+2Ch] [rbp-34h]

  v4 = *(_DWORD *)(a1 + 56);
  if ( v4 )
  {
    v12 = v4 - 1;
    v9 = 0;
    v38 = *(_QWORD *)(a1 + 40);
    v35 = sub_16704E0();
    v29 = sub_16704F0();
    v32 = 1;
    v40 = v12 & sub_16707B0(a2);
    while ( 1 )
    {
      v13 = (__int64 *)(v38 + 8LL * v40);
      if ( sub_1670560(a2, *v13) )
        goto LABEL_12;
      if ( sub_1670560(*v13, v35) )
        break;
      v25 = sub_1670560(*v13, v29);
      if ( !v9 && v25 )
        v9 = (__int64 *)(v38 + 8LL * v40);
      v40 = v12 & (v32 + v40);
      ++v32;
    }
    v21 = *(_DWORD *)(a1 + 48);
    v4 = *(_DWORD *)(a1 + 56);
    v5 = a1 + 32;
    if ( !v9 )
      v9 = (__int64 *)(v38 + 8LL * v40);
    ++*(_QWORD *)(a1 + 32);
    v10 = v21 + 1;
    if ( 4 * v10 < 3 * v4 )
    {
      if ( v4 - (v10 + *(_DWORD *)(a1 + 52)) > v4 >> 3 )
        goto LABEL_7;
      sub_16735E0(a1 + 32, v4);
      v22 = *(_DWORD *)(a1 + 56);
      if ( v22 )
      {
        v23 = *(_QWORD *)(a1 + 40);
        v24 = v22 - 1;
        v33 = sub_16704E0();
        v30 = sub_16704F0();
        v36 = 1;
        v37 = 0;
        v41 = v24 & sub_16707B0(a2);
        while ( 1 )
        {
          v9 = (__int64 *)(v23 + 8LL * v41);
          if ( sub_1670560(a2, *v9) )
            goto LABEL_6;
          if ( sub_1670560(*v9, v33) )
          {
LABEL_29:
            v10 = *(_DWORD *)(a1 + 48) + 1;
            if ( v37 )
              v9 = v37;
            goto LABEL_7;
          }
          v27 = sub_1670560(*v9, v30);
          if ( v37 || !v27 )
            v9 = v37;
          v37 = v9;
          v41 = v24 & (v36 + v41);
          ++v36;
        }
      }
LABEL_45:
      ++*(_DWORD *)(a1 + 48);
      sub_16704E0();
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 32);
    v5 = a1 + 32;
  }
  sub_16735E0(v5, 2 * v4);
  v6 = *(_DWORD *)(a1 + 56);
  if ( !v6 )
    goto LABEL_45;
  v7 = *(_QWORD *)(a1 + 40);
  v8 = v6 - 1;
  v31 = sub_16704E0();
  v28 = sub_16704F0();
  v34 = 1;
  v37 = 0;
  v39 = v8 & sub_16707B0(a2);
  while ( 1 )
  {
    v9 = (__int64 *)(v7 + 8LL * v39);
    if ( sub_1670560(a2, *v9) )
      break;
    if ( sub_1670560(*v9, v31) )
      goto LABEL_29;
    v26 = sub_1670560(*v9, v28);
    if ( v37 || !v26 )
      v9 = v37;
    v37 = v9;
    v39 = v8 & (v34 + v39);
    ++v34;
  }
LABEL_6:
  v10 = *(_DWORD *)(a1 + 48) + 1;
LABEL_7:
  *(_DWORD *)(a1 + 48) = v10;
  v11 = sub_16704E0();
  if ( !sub_1670560(*v9, v11) )
    --*(_DWORD *)(a1 + 52);
  *v9 = a2;
LABEL_12:
  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v15 = result - 1;
    v16 = *(_QWORD *)(a1 + 8);
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v17 = (__int64 *)(v16 + 8 * result);
    v18 = *v17;
    if ( *v17 == a2 )
    {
LABEL_14:
      *v17 = -16;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v19 = 1;
      while ( v18 != -8 )
      {
        v20 = v19 + 1;
        result = v15 & (unsigned int)(v19 + result);
        v17 = (__int64 *)(v16 + 8LL * (unsigned int)result);
        v18 = *v17;
        if ( *v17 == a2 )
          goto LABEL_14;
        v19 = v20;
      }
    }
  }
  return result;
}
