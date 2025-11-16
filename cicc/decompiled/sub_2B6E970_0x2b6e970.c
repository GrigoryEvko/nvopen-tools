// Function: sub_2B6E970
// Address: 0x2b6e970
//
__int64 __fastcall sub_2B6E970(__int64 a1, __int64 a2, int a3, __int64 a4, unsigned int a5)
{
  unsigned int v5; // r14d
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  int v12; // edi
  unsigned int v13; // r9d
  __int64 v14; // rdx
  __int64 v15; // r10
  __int64 v16; // rax
  int v17; // r10d
  unsigned int v18; // r9d
  __int64 v19; // rdx
  __int64 v20; // r11
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rdi
  unsigned int v27; // eax
  int v28; // r10d
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // rdx
  int v34; // r11d
  int v35; // edx
  int v36; // r12d
  __int64 v37; // rbx
  __int64 v38; // r12
  unsigned __int8 *v39; // rcx
  __int64 v40; // rax
  __int64 *v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // [rsp+8h] [rbp-68h]
  __int64 *v44; // [rsp+10h] [rbp-60h]
  __int64 v45; // [rsp+18h] [rbp-58h]
  __int64 v46[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v47[8]; // [rsp+30h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(**(_QWORD **)a1 + 8LL * a3);
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_QWORD *)(v9 + 8);
  v11 = *(unsigned int *)(v9 + 24);
  if ( (_DWORD)v11 )
  {
    v12 = v11 - 1;
    v13 = (v11 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v14 = v10 + 16LL * v13;
    v15 = *(_QWORD *)v14;
    if ( v8 == *(_QWORD *)v14 )
    {
LABEL_3:
      v16 = v10 + 16 * v11;
      if ( v14 != v16 )
      {
        v17 = *(_DWORD *)(v14 + 8);
        v18 = v12 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v19 = v10 + 16LL * v18;
        v20 = *(_QWORD *)v19;
        if ( a2 != *(_QWORD *)v19 )
        {
          v35 = 1;
          while ( v20 != -4096 )
          {
            v36 = v35 + 1;
            v18 = v12 & (v35 + v18);
            v19 = v10 + 16LL * v18;
            v20 = *(_QWORD *)v19;
            if ( a2 == *(_QWORD *)v19 )
              goto LABEL_5;
            v35 = v36;
          }
          v19 = v16;
        }
LABEL_5:
        LOBYTE(v5) = *(_DWORD *)(v19 + 8) == v17;
        if ( v8 == a2 )
          return 0;
        goto LABEL_11;
      }
    }
    else
    {
      v22 = 1;
      while ( v15 != -4096 )
      {
        v34 = v22 + 1;
        v13 = v12 & (v22 + v13);
        v14 = v10 + 16LL * v13;
        v15 = *(_QWORD *)v14;
        if ( v8 == *(_QWORD *)v14 )
          goto LABEL_3;
        v22 = v34;
      }
    }
  }
  v5 = 0;
  if ( v8 == a2 )
    return 0;
LABEL_11:
  if ( *(_BYTE *)v8 <= 0x1Cu )
    return 0;
  v23 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)v23 )
    return 0;
  v24 = *(_QWORD *)(v23 + 8);
  if ( (*(_BYTE *)(v24 + 88) & 1) != 0 )
  {
    v26 = v24 + 96;
    v25 = 3;
  }
  else
  {
    v25 = *(unsigned int *)(v24 + 104);
    v26 = *(_QWORD *)(v24 + 96);
    if ( !(_DWORD)v25 )
      goto LABEL_22;
    v25 = (unsigned int)(v25 - 1);
  }
  v27 = v25 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v10 = *(_QWORD *)(v26 + 72LL * v27);
  if ( v8 == v10 )
    return 0;
  v28 = 1;
  while ( v10 != -4096 )
  {
    v27 = v25 & (v28 + v27);
    v10 = *(_QWORD *)(v26 + 72LL * v27);
    if ( v8 == v10 )
      return 0;
    ++v28;
  }
LABEL_22:
  if ( (unsigned __int8)sub_2B15E10((char *)v8, v10, v25, a4, a5) )
    return 0;
  if ( (unsigned __int8)sub_2B2BA00(v31, v8, *(_QWORD *)(v31 + 3272), v29, v30) )
    return 0;
  LOBYTE(v5) = sub_2B14CA0(v8) ^ 1 | v5;
  if ( (_BYTE)v5 )
    return 0;
  v32 = *(_QWORD *)(a1 + 24);
  v46[0] = a2;
  v46[1] = v8;
  if ( !sub_2B5F980(v46, 2u, *(__int64 **)(v32 + 3304)) || !v33 || *(_QWORD *)(v8 + 40) != *(_QWORD *)(a2 + 40) )
    return 0;
  if ( *(_BYTE *)v8 == 84 )
  {
    v44 = *(__int64 **)(a1 + 32);
    if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 0 )
    {
      v37 = 0;
      v43 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      do
      {
        v38 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + v37);
        v45 = *(_QWORD *)(*(_QWORD *)(v8 - 8) + v37);
        if ( !(unsigned __int8)sub_2B0D8B0((unsigned __int8 *)v38) || !(unsigned __int8)sub_2B0D8B0(v39) )
        {
          v40 = *v44;
          v47[0] = v38;
          v41 = *(__int64 **)(v40 + 3304);
          v47[1] = v45;
          if ( !sub_2B5F980(v47, 2u, v41) || !v42 )
            return v5;
          if ( *(_QWORD *)(v38 + 40) != *(_QWORD *)(v45 + 40) )
            return 0;
        }
        v37 += 32;
      }
      while ( v37 != v43 );
    }
  }
  return 1;
}
