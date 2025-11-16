// Function: sub_2AB3940
// Address: 0x2ab3940
//
__int64 __fastcall sub_2AB3940(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  int v5; // eax
  int v7; // esi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // rdi
  unsigned int *v11; // r14
  __int64 v12; // r13
  __int64 v13; // rbx
  char v14; // r10
  __int64 v15; // rcx
  __int64 v16; // r12
  unsigned int *v17; // rbx
  unsigned int v18; // r14d
  char v19; // r15
  __int64 v20; // rsi
  __int64 v21; // r13
  unsigned int v22; // edx
  __int64 v23; // r9
  unsigned int v24; // esi
  unsigned int v25; // edx
  unsigned int v26; // edi
  int *v27; // rax
  int v28; // r11d
  __int64 v29; // rax
  __int64 result; // rax
  int v31; // eax
  __int64 v32; // rdx
  unsigned int v33; // eax
  __int64 v34; // r9
  int v35; // edi
  unsigned int v36; // eax
  unsigned int v37; // esi
  unsigned int v38; // ecx
  int *v39; // rax
  int v40; // r10d
  int v41; // eax
  int v42; // r8d
  int v43; // eax
  int v44; // r8d
  __int64 v45; // [rsp+8h] [rbp-58h]
  __int64 v47; // [rsp+18h] [rbp-48h]
  unsigned int v48; // [rsp+20h] [rbp-40h]
  char v49; // [rsp+2Ch] [rbp-34h]

  v49 = BYTE4(a3);
  v3 = *(_QWORD *)(a1 + 504);
  v4 = *(_QWORD *)(v3 + 56);
  v5 = *(_DWORD *)(v3 + 72);
  if ( !v5 )
    goto LABEL_70;
  v7 = v5 - 1;
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v4 + 16LL * v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    v41 = 1;
    while ( v10 != -4096 )
    {
      v42 = v41 + 1;
      v8 = v7 & (v41 + v8);
      v9 = (__int64 *)(v4 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      v41 = v42;
    }
LABEL_70:
    BUG();
  }
LABEL_3:
  v11 = (unsigned int *)v9[1];
  v48 = *v11;
  v12 = sub_B43CC0(a2);
  if ( *(_BYTE *)a2 == 61 )
    v13 = *(_QWORD *)(a2 + 8);
  else
    v13 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  if ( (unsigned __int8)sub_2AAE050(v13, v12) )
    return 0;
  v14 = v49;
  if ( !v49 )
  {
    if ( *(_BYTE *)(v13 + 8) != 14 )
    {
LABEL_11:
      v15 = v48;
      if ( !v48 )
        goto LABEL_37;
      goto LABEL_12;
    }
LABEL_10:
    v14 = *((_BYTE *)sub_AE2980(v12, *(_DWORD *)(v13 + 8) >> 8) + 16);
    goto LABEL_11;
  }
  v15 = v48;
  if ( !v48 || ((v48 - 1) & v48) != 0 )
    return 0;
  if ( *(_BYTE *)(v13 + 8) == 14 )
    goto LABEL_10;
  v14 = 0;
LABEL_12:
  v45 = a2;
  v16 = v13;
  v17 = v11;
  v47 = v12;
  v18 = 0;
  v19 = v14;
  do
  {
    v22 = v17[8];
    v23 = *((_QWORD *)v17 + 2);
    v24 = v18 + v17[10];
    if ( v22 )
    {
      v25 = v22 - 1;
      v26 = v25 & (37 * v24);
      v27 = (int *)(v23 + 16LL * v26);
      v28 = *v27;
      if ( *v27 == v24 )
      {
LABEL_23:
        v29 = *((_QWORD *)v27 + 1);
        if ( v29 )
        {
          if ( *(_BYTE *)v29 != 61 )
            v29 = *(_QWORD *)(v29 - 64);
          v21 = *(_QWORD *)(v29 + 8);
          if ( *(_BYTE *)(v21 + 8) == 14 && *((_BYTE *)sub_AE2980(v47, *(_DWORD *)(v21 + 8) >> 8) + 16) )
          {
            if ( !v19 )
              return 0;
            v20 = v16;
            if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
              v20 = **(_QWORD **)(v16 + 16);
            if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 <= 1 )
              v21 = **(_QWORD **)(v21 + 16);
            if ( *(_DWORD *)(v21 + 8) >> 8 != *(_DWORD *)(v20 + 8) >> 8 )
              return 0;
          }
          else if ( v19 )
          {
            return 0;
          }
        }
      }
      else
      {
        v31 = 1;
        while ( v28 != 0x7FFFFFFF )
        {
          v15 = (unsigned int)(v31 + 1);
          v26 = v25 & (v31 + v26);
          v27 = (int *)(v23 + 16LL * v26);
          v28 = *v27;
          if ( v24 == *v27 )
            goto LABEL_23;
          v31 = v15;
        }
      }
    }
    ++v18;
  }
  while ( v18 < v48 );
  a2 = v45;
  v11 = v17;
LABEL_37:
  if ( *(_BYTE *)(a1 + 108) && (v32 = *(unsigned int *)(a1 + 100), (_DWORD)v32)
    || (v33 = sub_31A6C30(*(_QWORD *)(a1 + 440), *(_QWORD *)(a2 + 40)), v32 = v33, (_BYTE)v33) )
  {
    LOBYTE(v32) = sub_B19060(*(_QWORD *)(a1 + 440) + 440LL, a2, v32, v15);
    if ( *(_BYTE *)a2 != 61 )
    {
      if ( (*(_BYTE *)a2 != 62 || *v11 <= v11[6]) && !(_BYTE)v32 )
        return 1;
LABEL_42:
      if ( !*((_BYTE *)v11 + 4) )
      {
        sub_2AAE0E0(a2);
        return sub_DFA360(*(_QWORD *)(a1 + 448));
      }
      return 0;
    }
  }
  else if ( *(_BYTE *)a2 != 61 )
  {
    if ( *(_BYTE *)a2 != 62 || *v11 <= v11[6] )
      return 1;
    goto LABEL_42;
  }
  v34 = *((_QWORD *)v11 + 2);
  v35 = *v11 + v11[10] - 1;
  v36 = v11[8];
  if ( v36 )
  {
    v37 = v36 - 1;
    v38 = (v36 - 1) & (37 * v35);
    v39 = (int *)(v34 + 16LL * v38);
    v40 = *v39;
    if ( *v39 == v35 )
    {
LABEL_54:
      if ( *((_QWORD *)v39 + 1) )
      {
        if ( !(_BYTE)v32 )
          return 1;
LABEL_56:
        if ( !*((_BYTE *)v11 + 4) )
        {
          sub_2AAE0E0(a2);
          return sub_DFA390(*(_QWORD *)(a1 + 448));
        }
        return 0;
      }
    }
    else
    {
      v43 = 1;
      while ( v40 != 0x7FFFFFFF )
      {
        v44 = v43 + 1;
        v38 = v37 & (v43 + v38);
        v39 = (int *)(v34 + 16LL * v38);
        v40 = *v39;
        if ( v35 == *v39 )
          goto LABEL_54;
        v43 = v44;
      }
    }
  }
  if ( *(_DWORD *)(a1 + 96) )
    goto LABEL_56;
  result = 1;
  if ( (_BYTE)v32 )
    goto LABEL_56;
  return result;
}
