// Function: sub_19114F0
// Address: 0x19114f0
//
__int64 __fastcall sub_19114F0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // r14d
  unsigned int v6; // esi
  __int64 v8; // rdx
  __int64 v9; // rdi
  int v10; // r11d
  __int64 *v11; // r10
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r9
  char v15; // dl
  int v16; // eax
  int v17; // ecx
  __int64 v18; // r12
  _QWORD *v19; // rax
  unsigned int v20; // r13d
  unsigned int v21; // esi
  __int64 v22; // rcx
  __int64 v23; // r9
  unsigned int v24; // edx
  __int64 v25; // rax
  __int64 v26; // r10
  int v27; // r11d
  __int64 v28; // rdi
  int v29; // eax
  int v30; // edx
  _QWORD *v31; // r15
  unsigned int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // rdi
  unsigned int v35; // ecx
  __int64 *v36; // rdx
  __int64 v37; // r11
  unsigned __int64 v38; // rdi
  int v39; // r12d
  unsigned __int64 v40; // rax
  int v41; // r8d
  int v42; // r9d
  __int64 v43; // rdx
  __int64 v44; // r13
  __int64 *v45; // r14
  unsigned int i; // r15d
  __int64 v47; // rax
  int v48; // r13d
  __int64 *v49; // r10
  int v50; // eax
  int v51; // eax
  int v52; // [rsp+10h] [rbp-160h]
  __int64 v53; // [rsp+18h] [rbp-158h] BYREF
  __int64 v54; // [rsp+20h] [rbp-150h] BYREF
  __int64 *v55; // [rsp+28h] [rbp-148h] BYREF
  _QWORD *v56; // [rsp+30h] [rbp-140h] BYREF
  __int64 v57; // [rsp+38h] [rbp-138h]
  _QWORD v58[38]; // [rsp+40h] [rbp-130h] BYREF

  v3 = 0;
  v53 = a1;
  if ( a3 > dword_4FAE6C0 )
    return v3;
  v6 = *(_DWORD *)(a2 + 24);
  v56 = (_QWORD *)a1;
  LOBYTE(v57) = 2;
  v8 = a1;
  if ( v6 )
  {
    v9 = *(_QWORD *)(a2 + 8);
    v10 = 1;
    v11 = 0;
    v12 = (v6 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
    {
LABEL_5:
      v15 = *((_BYTE *)v13 + 8);
      if ( v15 == 2 )
      {
        *((_BYTE *)v13 + 8) = 3;
        return 1;
      }
      else
      {
        LOBYTE(v3) = v15 != 0;
      }
      return v3;
    }
    while ( v14 != -8 )
    {
      if ( v14 == -16 && !v11 )
        v11 = v13;
      v12 = (v6 - 1) & (v10 + v12);
      v13 = (__int64 *)(v9 + 16LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
        goto LABEL_5;
      ++v10;
    }
    if ( !v11 )
      v11 = v13;
    v16 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v17 = v16 + 1;
    if ( 4 * (v16 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a2 + 20) - v17 > v6 >> 3 )
        goto LABEL_18;
      goto LABEL_75;
    }
  }
  else
  {
    ++*(_QWORD *)a2;
  }
  v6 *= 2;
LABEL_75:
  sub_1911330(a2, v6);
  sub_190EEA0(a2, (__int64 *)&v56, &v55);
  v11 = v55;
  v8 = (__int64)v56;
  v17 = *(_DWORD *)(a2 + 16) + 1;
LABEL_18:
  *(_DWORD *)(a2 + 16) = v17;
  if ( *v11 != -8 )
    --*(_DWORD *)(a2 + 20);
  *v11 = v8;
  *((_BYTE *)v11 + 8) = v57;
  v18 = *(_QWORD *)(v53 + 8);
  if ( v18 )
  {
    while ( 1 )
    {
      v19 = sub_1648700(v18);
      if ( (unsigned __int8)(*((_BYTE *)v19 + 16) - 25) <= 9u )
        break;
      v18 = *(_QWORD *)(v18 + 8);
      if ( !v18 )
        goto LABEL_26;
    }
    v20 = a3 + 1;
    while ( 1 )
    {
      v3 = sub_19114F0(v19[5], a2, v20);
      if ( !(_BYTE)v3 )
        break;
      do
      {
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          return v3;
        v19 = sub_1648700(v18);
      }
      while ( (unsigned __int8)(*((_BYTE *)v19 + 16) - 25) > 9u );
    }
  }
LABEL_26:
  v21 = *(_DWORD *)(a2 + 24);
  if ( !v21 )
  {
    ++*(_QWORD *)a2;
LABEL_77:
    v21 *= 2;
    goto LABEL_78;
  }
  v22 = v53;
  v23 = *(_QWORD *)(a2 + 8);
  v24 = (v21 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
  v25 = v23 + 16LL * v24;
  v26 = *(_QWORD *)v25;
  if ( *(_QWORD *)v25 == v53 )
  {
LABEL_28:
    if ( *(_BYTE *)(v25 + 8) == 2 )
    {
      *(_BYTE *)(v25 + 8) = 0;
      return 0;
    }
    goto LABEL_41;
  }
  v27 = 1;
  v28 = 0;
  while ( v26 != -8 )
  {
    if ( !v28 && v26 == -16 )
      v28 = v25;
    v24 = (v21 - 1) & (v27 + v24);
    v25 = v23 + 16LL * v24;
    v26 = *(_QWORD *)v25;
    if ( v53 == *(_QWORD *)v25 )
      goto LABEL_28;
    ++v27;
  }
  if ( !v28 )
    v28 = v25;
  v29 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v30 = v29 + 1;
  if ( 4 * (v29 + 1) >= 3 * v21 )
    goto LABEL_77;
  if ( v21 - *(_DWORD *)(a2 + 20) - v30 <= v21 >> 3 )
  {
LABEL_78:
    sub_1911330(a2, v21);
    sub_190EEA0(a2, &v53, &v56);
    v28 = (__int64)v56;
    v22 = v53;
    v30 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v30;
  if ( *(_QWORD *)v28 != -8 )
    --*(_DWORD *)(a2 + 20);
  *(_QWORD *)v28 = v22;
  *(_BYTE *)(v28 + 8) = 0;
  v23 = *(_QWORD *)(a2 + 8);
  v21 = *(_DWORD *)(a2 + 24);
LABEL_41:
  v31 = v58;
  v56 = v58;
  v58[0] = v53;
  v57 = 0x2000000001LL;
  v32 = 1;
  while ( 1 )
  {
    v33 = v32--;
    v34 = v31[v33 - 1];
    LODWORD(v57) = v32;
    v54 = v34;
    if ( !v21 )
    {
      ++*(_QWORD *)a2;
LABEL_71:
      v21 *= 2;
LABEL_72:
      sub_1911330(a2, v21);
      sub_190EEA0(a2, &v54, &v55);
      v49 = v55;
      v34 = v54;
      v51 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_67;
    }
    v35 = (v21 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
    v36 = (__int64 *)(v23 + 16LL * v35);
    v37 = *v36;
    if ( v34 == *v36 )
    {
LABEL_44:
      if ( *((_BYTE *)v36 + 8) )
      {
        *((_BYTE *)v36 + 8) = 0;
        v38 = sub_157EBA0(v34);
        if ( v38 )
        {
          v39 = sub_15F4D60(v38);
          v40 = sub_157EBA0(v54);
          v43 = (unsigned int)v57;
          v31 = v56;
          v44 = v40;
          v52 = v39;
          if ( v39 > HIDWORD(v57) - (unsigned __int64)(unsigned int)v57 )
          {
            sub_16CD150((__int64)&v56, v58, v39 + (unsigned __int64)(unsigned int)v57, 8, v41, v42);
            v31 = v56;
            v43 = (unsigned int)v57;
          }
          v45 = &v31[v43];
          if ( v39 )
          {
            for ( i = 0; i != v39; ++i )
            {
              v47 = sub_15F4DF0(v44, i);
              if ( v45 )
                *v45 = v47;
              ++v45;
            }
            LODWORD(v43) = v57;
            v31 = v56;
          }
        }
        else
        {
          v52 = 0;
          LODWORD(v43) = v57;
        }
        LODWORD(v57) = v52 + v43;
        v32 = v52 + v43;
      }
      goto LABEL_55;
    }
    v48 = 1;
    v49 = 0;
    while ( v37 != -8 )
    {
      if ( v37 == -16 && !v49 )
        v49 = v36;
      v35 = (v21 - 1) & (v48 + v35);
      v36 = (__int64 *)(v23 + 16LL * v35);
      v37 = *v36;
      if ( v34 == *v36 )
        goto LABEL_44;
      ++v48;
    }
    v50 = *(_DWORD *)(a2 + 16);
    if ( !v49 )
      v49 = v36;
    ++*(_QWORD *)a2;
    v51 = v50 + 1;
    if ( 4 * v51 >= 3 * v21 )
      goto LABEL_71;
    if ( v21 - (v51 + *(_DWORD *)(a2 + 20)) <= v21 >> 3 )
      goto LABEL_72;
LABEL_67:
    *(_DWORD *)(a2 + 16) = v51;
    if ( *v49 != -8 )
      --*(_DWORD *)(a2 + 20);
    *v49 = v34;
    *((_BYTE *)v49 + 8) = 0;
    v32 = v57;
    v31 = v56;
LABEL_55:
    if ( !v32 )
      break;
    v23 = *(_QWORD *)(a2 + 8);
    v21 = *(_DWORD *)(a2 + 24);
  }
  if ( v31 != v58 )
    _libc_free((unsigned __int64)v31);
  return 0;
}
