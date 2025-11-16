// Function: sub_A58A30
// Address: 0xa58a30
//
__int64 __fastcall sub_A58A30(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  unsigned int v5; // esi
  unsigned int v6; // edi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 result; // rax
  unsigned __int8 v11; // dl
  unsigned int v12; // eax
  int v13; // eax
  int v14; // r13d
  int v15; // r9d
  __int64 *v16; // r14
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rax
  int v21; // eax
  int v22; // eax
  __int64 v23; // rax
  unsigned __int8 **v24; // r13
  unsigned __int8 **v25; // r14
  unsigned __int8 v26; // al
  int v27; // eax
  __int64 v28; // rdx
  int v29; // eax
  __int64 v30; // rax
  __int64 *v31; // rax
  int v32; // eax
  int v33; // r9d
  __int64 *v34; // [rsp+8h] [rbp-38h] BYREF
  __int64 v35; // [rsp+10h] [rbp-30h] BYREF
  int v36; // [rsp+18h] [rbp-28h]

  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_DWORD *)(a2 + 24);
  if ( !v5 )
  {
    v11 = *(_BYTE *)a1;
    if ( *(_BYTE *)a1 > 0x15u )
    {
      v27 = *(_DWORD *)(a2 + 40);
      v35 = a1;
      v36 = 0;
      v14 = v27 + 1;
      goto LABEL_29;
    }
    goto LABEL_18;
  }
  v6 = v5 - 1;
  v7 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v8 = (__int64 *)(v4 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != a1 )
  {
    v21 = 1;
    while ( v9 != -4096 )
    {
      v33 = v21 + 1;
      v7 = v6 & (v21 + v7);
      v8 = (__int64 *)(v4 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a1 )
        goto LABEL_3;
      v21 = v33;
    }
LABEL_15:
    v11 = *(_BYTE *)a1;
    if ( *(_BYTE *)a1 > 0x15u )
      goto LABEL_16;
LABEL_18:
    v12 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    if ( !v12 )
      goto LABEL_8;
    goto LABEL_7;
  }
LABEL_3:
  if ( v8 == (__int64 *)(v4 + 16LL * v5) )
    goto LABEL_15;
  result = *(unsigned int *)(*(_QWORD *)(a2 + 32) + 16LL * *((unsigned int *)v8 + 2) + 8);
  if ( (_DWORD)result )
    return result;
  v11 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 <= 0x15u )
  {
    v12 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    if ( v12 )
    {
LABEL_7:
      if ( v11 > 3u )
      {
        v23 = 4LL * v12;
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        {
          v24 = *(unsigned __int8 ***)(a1 - 8);
          v25 = &v24[v23];
        }
        else
        {
          v25 = (unsigned __int8 **)a1;
          v24 = (unsigned __int8 **)(a1 - v23 * 8);
        }
        do
        {
          v26 = **v24;
          if ( v26 != 23 && v26 > 3u )
            sub_A58A30(*v24, a2);
          v24 += 4;
        }
        while ( v25 != v24 );
        v4 = *(_QWORD *)(a2 + 8);
        v5 = *(_DWORD *)(a2 + 24);
      }
LABEL_8:
      v13 = *(_DWORD *)(a2 + 40);
      v35 = a1;
      v36 = 0;
      v14 = v13 + 1;
      if ( v5 )
      {
        v6 = v5 - 1;
        goto LABEL_10;
      }
LABEL_29:
      ++*(_QWORD *)a2;
      v5 = 0;
      v34 = 0;
      goto LABEL_30;
    }
  }
LABEL_16:
  v22 = *(_DWORD *)(a2 + 40);
  v35 = a1;
  v36 = 0;
  v14 = v22 + 1;
LABEL_10:
  v15 = 1;
  v16 = 0;
  v17 = v6 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v18 = (__int64 *)(v4 + 16LL * v17);
  v19 = *v18;
  if ( *v18 != a1 )
  {
    while ( v19 != -4096 )
    {
      if ( !v16 && v19 == -8192 )
        v16 = v18;
      v17 = v6 & (v15 + v17);
      v18 = (__int64 *)(v4 + 16LL * v17);
      v19 = *v18;
      if ( *v18 == a1 )
        goto LABEL_11;
      ++v15;
    }
    if ( !v16 )
      v16 = v18;
    v32 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v29 = v32 + 1;
    v34 = v16;
    if ( 4 * v29 < 3 * v5 )
    {
      v28 = a1;
      if ( v5 - (v29 + *(_DWORD *)(a2 + 20)) > v5 >> 3 )
        goto LABEL_32;
      goto LABEL_31;
    }
LABEL_30:
    v5 *= 2;
LABEL_31:
    sub_A429D0(a2, v5);
    sub_A56BF0(a2, &v35, &v34);
    v28 = v35;
    v16 = v34;
    v29 = *(_DWORD *)(a2 + 16) + 1;
LABEL_32:
    *(_DWORD *)(a2 + 16) = v29;
    if ( *v16 != -4096 )
      --*(_DWORD *)(a2 + 20);
    *v16 = v28;
    *((_DWORD *)v16 + 2) = v36;
    v30 = *(unsigned int *)(a2 + 40);
    if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 44) )
    {
      sub_C8D5F0(a2 + 32, a2 + 48, v30 + 1, 16);
      v30 = *(unsigned int *)(a2 + 40);
    }
    v31 = (__int64 *)(*(_QWORD *)(a2 + 32) + 16 * v30);
    *v31 = a1;
    v31[1] = 0;
    v20 = *(unsigned int *)(a2 + 40);
    *(_DWORD *)(a2 + 40) = v20 + 1;
    *((_DWORD *)v16 + 2) = v20;
    goto LABEL_12;
  }
LABEL_11:
  v20 = *((unsigned int *)v18 + 2);
LABEL_12:
  result = *(_QWORD *)(a2 + 32) + 16 * v20;
  *(_DWORD *)(result + 8) = v14;
  return result;
}
