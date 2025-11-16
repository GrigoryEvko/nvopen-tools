// Function: sub_3873B70
// Address: 0x3873b70
//
__int64 __fastcall sub_3873B70(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rcx
  __int64 v7; // r9
  __int64 *v8; // r15
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // r13
  int v14; // eax
  int v15; // edx
  __int64 v16; // rcx
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // rcx
  __int64 *v22; // r15
  __int64 v23; // rsi
  __int64 v24; // rax
  unsigned int v25; // esi
  __int64 v26; // rdi
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // r13
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  int v38; // r10d
  __int64 *v39; // r9
  int v40; // ecx
  int v41; // ecx
  __int64 *v42; // [rsp+0h] [rbp-60h]
  __int64 v43; // [rsp+8h] [rbp-58h]
  __int64 *v44; // [rsp+18h] [rbp-48h] BYREF
  __int64 *v45; // [rsp+20h] [rbp-40h] BYREF
  __int64 v46; // [rsp+28h] [rbp-38h]

  v2 = a1 + 120;
  v4 = a2;
  v45 = (__int64 *)a2;
  v5 = *(_DWORD *)(a1 + 144);
  v46 = 0;
  if ( v5 )
  {
    v6 = *(_QWORD *)(a1 + 128);
    v7 = 1;
    v8 = 0;
    v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
      return v10[1];
    while ( v11 != -8 )
    {
      if ( !v8 && v11 == -16 )
        v8 = v10;
      v9 = (v5 - 1) & (v7 + v9);
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( v4 == *v10 )
        return v10[1];
      v7 = (unsigned int)(v7 + 1);
    }
    if ( !v8 )
      v8 = v10;
    v14 = *(_DWORD *)(a1 + 136);
    ++*(_QWORD *)(a1 + 120);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v5 )
    {
      v16 = v4;
      if ( v5 - *(_DWORD *)(a1 + 140) - v15 > v5 >> 3 )
        goto LABEL_15;
      goto LABEL_34;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 120);
  }
  v5 *= 2;
LABEL_34:
  sub_3873880(v2, v5);
  sub_3872FC0(v2, (__int64 *)&v45, &v44);
  v8 = v44;
  v16 = (__int64)v45;
  v15 = *(_DWORD *)(a1 + 136) + 1;
LABEL_15:
  *(_DWORD *)(a1 + 136) = v15;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 140);
  *v8 = v16;
  v8[1] = v46;
  v17 = *(unsigned __int16 *)(v4 + 24);
  if ( !(_WORD)v17 )
    return 0;
  if ( (_WORD)v17 != 10 )
  {
    if ( (unsigned __int16)(v17 - 7) > 2u )
    {
      v20 = (unsigned int)(v17 - 4);
      if ( (unsigned __int16)(v17 - 4) > 1u )
      {
        if ( (unsigned __int16)(v17 - 1) > 2u )
        {
          v31 = *(_QWORD *)(*(_QWORD *)a1 + 56LL);
          v32 = sub_3873B70(a1, *(_QWORD *)(v4 + 40), v20, v16, v11, v7);
          v37 = sub_3873B70(a1, *(_QWORD *)(v4 + 32), v33, v34, v35, v36);
          v30 = sub_386EC30(v37, v32, v31);
        }
        else
        {
          v30 = sub_3873B70(a1, *(_QWORD *)(v4 + 32), v20, v16, v11, v7);
        }
        v45 = (__int64 *)v4;
        v12 = v30;
        v28 = sub_3873A40(v2, (__int64 *)&v45);
        goto LABEL_30;
      }
    }
    v12 = 0;
    if ( (_WORD)v17 == 7 )
      v12 = *(_QWORD *)(v4 + 48);
    v21 = *(__int64 **)(v4 + 32);
    v42 = &v21[*(_QWORD *)(v4 + 40)];
    if ( v21 != v42 )
    {
      v22 = *(__int64 **)(v4 + 32);
      do
      {
        v23 = *v22++;
        v43 = *(_QWORD *)(*(_QWORD *)a1 + 56LL);
        v24 = sub_3873B70(a1, v23, v43, v21, v11, v7);
        v12 = sub_386EC30(v12, v24, v43);
      }
      while ( v42 != v22 );
    }
    v25 = *(_DWORD *)(a1 + 144);
    v44 = (__int64 *)v4;
    if ( v25 )
    {
      v26 = *(_QWORD *)(a1 + 128);
      v27 = (v25 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( v4 == *v28 )
      {
LABEL_30:
        v28[1] = v12;
        return v12;
      }
      v38 = 1;
      v39 = 0;
      while ( v29 != -8 )
      {
        if ( !v39 && v29 == -16 )
          v39 = v28;
        v27 = (v25 - 1) & (v38 + v27);
        v28 = (__int64 *)(v26 + 16LL * v27);
        v29 = *v28;
        if ( v4 == *v28 )
          goto LABEL_30;
        ++v38;
      }
      v40 = *(_DWORD *)(a1 + 136);
      if ( v39 )
        v28 = v39;
      ++*(_QWORD *)(a1 + 120);
      v41 = v40 + 1;
      if ( 4 * v41 < 3 * v25 )
      {
        if ( v25 - *(_DWORD *)(a1 + 140) - v41 > v25 >> 3 )
        {
LABEL_45:
          *(_DWORD *)(a1 + 136) = v41;
          if ( *v28 != -8 )
            --*(_DWORD *)(a1 + 140);
          *v28 = v4;
          v28[1] = 0;
          goto LABEL_30;
        }
LABEL_50:
        sub_3873880(v2, v25);
        sub_3872FC0(v2, (__int64 *)&v44, &v45);
        v28 = v45;
        v4 = (__int64)v44;
        v41 = *(_DWORD *)(a1 + 136) + 1;
        goto LABEL_45;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 120);
    }
    v25 *= 2;
    goto LABEL_50;
  }
  v18 = *(_QWORD *)(v4 - 8);
  v12 = 0;
  if ( *(_BYTE *)(v18 + 16) > 0x17u )
  {
    v19 = sub_13AE450(*(_QWORD *)(*(_QWORD *)a1 + 64LL), *(_QWORD *)(v18 + 40));
    v8[1] = v19;
    return v19;
  }
  return v12;
}
