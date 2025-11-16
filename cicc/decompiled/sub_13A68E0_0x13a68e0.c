// Function: sub_13A68E0
// Address: 0x13a68e0
//
__int64 __fastcall sub_13A68E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 result; // rax
  int v7; // r15d
  __int64 v8; // r11
  __int64 v9; // r14
  unsigned int v10; // r9d
  __int64 *v11; // rcx
  __int64 v12; // r12
  __int64 *v13; // rax
  _QWORD **v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rbx
  unsigned int v17; // r10d
  __int64 *v18; // rsi
  __int64 v19; // r13
  __int64 *v20; // rdx
  _QWORD **v21; // rdx
  _QWORD *v22; // r8
  unsigned int v23; // edx
  int v24; // r8d
  _QWORD *v25; // rcx
  _QWORD *v26; // rsi
  __int64 v27; // r8
  int v28; // edx
  int v29; // ecx
  __int64 v30; // rdx
  unsigned int v31; // r8d
  int v32; // eax
  int v33; // esi
  int v34; // r8d
  __int64 v35; // rdx
  int v36; // r10d
  int v37; // r9d
  __int64 v38; // [rsp-8h] [rbp-8h]

  v5 = *(_QWORD *)(a1 + 16);
  result = *(unsigned int *)(v5 + 24);
  if ( !(_DWORD)result )
  {
    *(_DWORD *)(a1 + 36) = 0;
    *(_DWORD *)(a1 + 32) = 0;
    *(_DWORD *)(a1 + 40) = 0;
    return result;
  }
  v7 = result - 1;
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(v5 + 8);
  v10 = (result - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v11 = (__int64 *)(v9 + 16LL * v10);
  v12 = *v11;
  v13 = v11;
  if ( v8 != *v11 )
  {
    v30 = *v11;
    v31 = v10;
    v32 = 1;
    while ( v30 != -8 )
    {
      v36 = v32 + 1;
      v31 = v7 & (v32 + v31);
      v13 = (__int64 *)(v9 + 16LL * v31);
      v30 = *v13;
      if ( v8 == *v13 )
        goto LABEL_3;
      v32 = v36;
    }
    goto LABEL_33;
  }
LABEL_3:
  v14 = (_QWORD **)v13[1];
  if ( !v14 )
  {
LABEL_33:
    result = 0;
    goto LABEL_6;
  }
  v15 = *v14;
  for ( result = 1; v15; result = (unsigned int)(result + 1) )
    v15 = (_QWORD *)*v15;
LABEL_6:
  v16 = *(_QWORD *)(a3 + 40);
  v17 = v7 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
  v18 = (__int64 *)(v9 + 16LL * v17);
  v19 = *v18;
  v20 = v18;
  if ( v16 == *v18 )
  {
LABEL_7:
    v21 = (_QWORD **)v20[1];
    if ( v21 )
    {
      v22 = *v21;
      v23 = 1;
      if ( v22 )
      {
        do
        {
          v22 = (_QWORD *)*v22;
          ++v23;
        }
        while ( v22 );
        v24 = v23 + result;
      }
      else
      {
        v24 = result + 1;
      }
      goto LABEL_11;
    }
  }
  else
  {
    *((_DWORD *)&v38 - 11) = v17;
    v27 = v19;
    v28 = 1;
    while ( v27 != -8 )
    {
      v34 = v28 + 1;
      v35 = v7 & (unsigned int)(*((_DWORD *)&v38 - 11) + v28);
      *((_DWORD *)&v38 - 12) = v34;
      *((_DWORD *)&v38 - 11) = v35;
      v20 = (__int64 *)(v9 + 16 * v35);
      v27 = *v20;
      if ( v16 == *v20 )
        goto LABEL_7;
      v28 = *((_DWORD *)&v38 - 12);
    }
  }
  v24 = result;
  v23 = 0;
LABEL_11:
  if ( v8 == v12 )
  {
LABEL_12:
    v25 = (_QWORD *)v11[1];
  }
  else
  {
    v29 = 1;
    while ( v12 != -8 )
    {
      v10 = v7 & (v29 + v10);
      *((_DWORD *)&v38 - 11) = v29 + 1;
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v8 == *v11 )
        goto LABEL_12;
      v29 = *((_DWORD *)&v38 - 11);
    }
    v25 = 0;
  }
  if ( v16 == v19 )
  {
LABEL_14:
    v26 = (_QWORD *)v18[1];
  }
  else
  {
    v33 = 1;
    while ( v19 != -8 )
    {
      v37 = v33 + 1;
      v17 = v7 & (v33 + v17);
      v18 = (__int64 *)(v9 + 16LL * v17);
      v19 = *v18;
      if ( v16 == *v18 )
        goto LABEL_14;
      v33 = v37;
    }
    v26 = 0;
  }
  *(_DWORD *)(a1 + 36) = result;
  *(_DWORD *)(a1 + 40) = v24;
  if ( v23 >= (unsigned int)result )
  {
    if ( v23 > (unsigned int)result )
    {
      do
      {
        --v23;
        v26 = (_QWORD *)*v26;
      }
      while ( v23 != (_DWORD)result );
    }
  }
  else
  {
    do
    {
      result = (unsigned int)(result - 1);
      v25 = (_QWORD *)*v25;
    }
    while ( (_DWORD)result != v23 );
  }
  for ( ; v26 != v25; result = (unsigned int)(result - 1) )
  {
    v25 = (_QWORD *)*v25;
    v26 = (_QWORD *)*v26;
  }
  *(_DWORD *)(a1 + 32) = result;
  *(_DWORD *)(a1 + 40) = v24 - result;
  return result;
}
