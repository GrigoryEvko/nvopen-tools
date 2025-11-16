// Function: sub_2D595B0
// Address: 0x2d595b0
//
__int64 *__fastcall sub_2D595B0(__int64 a1)
{
  _QWORD *v2; // r12
  __int64 v3; // r12
  unsigned int *v4; // rcx
  unsigned int *i; // r8
  __int64 *v6; // r13
  __int64 *v7; // r14
  __int64 v8; // rdi
  __int64 *v9; // r13
  __int64 *v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r12
  __int64 v28; // rsi
  __int64 *v29; // rdi
  __int64 *v30; // rdx
  __int64 *result; // rax
  __int64 v32; // rcx
  __int16 v33; // dx
  unsigned __int64 *v34; // r8
  unsigned __int8 v35; // al
  char v36; // dl
  __int64 v37; // rsi
  __int64 v38; // rcx

  v2 = *(_QWORD **)(a1 + 8);
  if ( *(_BYTE *)(a1 + 56) )
  {
    if ( v2[5] )
      sub_B43D10(*(_QWORD **)(a1 + 8));
    sub_B43E90((__int64)v2, *(_QWORD *)(a1 + 16));
  }
  else
  {
    v34 = (unsigned __int64 *)sub_AA5190(*(_QWORD *)(a1 + 32));
    if ( v34 )
    {
      v35 = v33;
      v36 = HIBYTE(v33);
    }
    else
    {
      v36 = 0;
      v35 = 0;
    }
    v37 = *(_QWORD *)(a1 + 32);
    v38 = v35;
    BYTE1(v38) = v36;
    if ( v2[5] )
      sub_B44550(v2, v37, v34, v38);
    else
      sub_B44150(v2, v37, v34, v38);
  }
  sub_AA61A0(v2[5], (__int64)v2, *(_QWORD *)(a1 + 40), *(_QWORD *)(a1 + 48));
  v3 = *(_QWORD *)(a1 + 128);
  if ( v3 )
  {
    v4 = *(unsigned int **)(v3 + 16);
    for ( i = &v4[4 * *(unsigned int *)(v3 + 24)]; i != v4; v4 += 4 )
    {
      v25 = *(_QWORD *)v4;
      v26 = *(_QWORD *)(v3 + 8);
      if ( (*(_BYTE *)(*(_QWORD *)v4 + 7LL) & 0x40) != 0 )
        v21 = *(_QWORD *)(v25 - 8);
      else
        v21 = v25 - 32LL * (*(_DWORD *)(v25 + 4) & 0x7FFFFFF);
      v22 = 32LL * v4[2] + v21;
      if ( *(_QWORD *)v22 )
      {
        v23 = *(_QWORD *)(v22 + 8);
        **(_QWORD **)(v22 + 16) = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
      }
      *(_QWORD *)v22 = v26;
      if ( v26 )
      {
        v24 = *(_QWORD *)(v26 + 16);
        *(_QWORD *)(v22 + 8) = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 16) = v22 + 8;
        *(_QWORD *)(v22 + 16) = v26 + 16;
        *(_QWORD *)(v26 + 16) = v22;
      }
    }
    v6 = *(__int64 **)(v3 + 96);
    v7 = &v6[*(unsigned int *)(v3 + 104)];
    while ( v7 != v6 )
    {
      v8 = *v6++;
      sub_B59720(v8, *(_QWORD *)(v3 + 144), *(unsigned __int8 **)(v3 + 8));
    }
    v9 = *(__int64 **)(v3 + 120);
    v10 = &v9[*(unsigned int *)(v3 + 128)];
    while ( v10 != v9 )
    {
      v11 = *v9++;
      sub_B13360(v11, *(unsigned __int8 **)(v3 + 144), *(unsigned __int8 **)(v3 + 8), 0);
    }
  }
  v12 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v12 )
  {
    v13 = 8 * v12;
    v14 = 0;
    do
    {
      v19 = *(_QWORD *)(a1 + 72);
      v20 = *(_QWORD *)(*(_QWORD *)(a1 + 80) + v14);
      if ( (*(_BYTE *)(v19 + 7) & 0x40) != 0 )
        v15 = *(_QWORD *)(v19 - 8);
      else
        v15 = v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF);
      v16 = v15 + 4 * v14;
      if ( *(_QWORD *)v16 )
      {
        v17 = *(_QWORD *)(v16 + 8);
        **(_QWORD **)(v16 + 16) = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = *(_QWORD *)(v16 + 16);
      }
      *(_QWORD *)v16 = v20;
      if ( v20 )
      {
        v18 = *(_QWORD *)(v20 + 16);
        *(_QWORD *)(v16 + 8) = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = v16 + 8;
        *(_QWORD *)(v16 + 16) = v20 + 16;
        *(_QWORD *)(v20 + 16) = v16;
      }
      v14 += 8;
    }
    while ( v14 != v13 );
  }
  v27 = *(_QWORD *)(a1 + 136);
  v28 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v27 + 28) )
  {
    v29 = *(__int64 **)(v27 + 8);
    v30 = &v29[*(unsigned int *)(v27 + 20)];
    result = v29;
    if ( v29 != v30 )
    {
      while ( v28 != *result )
      {
        if ( v30 == ++result )
          return result;
      }
      v32 = (unsigned int)(*(_DWORD *)(v27 + 20) - 1);
      *(_DWORD *)(v27 + 20) = v32;
      *result = v29[v32];
      ++*(_QWORD *)v27;
    }
  }
  else
  {
    result = sub_C8CA60(*(_QWORD *)(a1 + 136), v28);
    if ( result )
    {
      *result = -2;
      ++*(_DWORD *)(v27 + 24);
      ++*(_QWORD *)v27;
    }
  }
  return result;
}
