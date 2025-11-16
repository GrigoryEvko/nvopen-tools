// Function: sub_1A55C50
// Address: 0x1a55c50
//
__int64 __fastcall sub_1A55C50(_QWORD *a1, __int64 a2, __int64 a3)
{
  char v6; // cl
  __int64 v7; // rdx
  int v8; // esi
  __int64 v9; // r8
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // r9
  __int64 v13; // rsi
  unsigned int v14; // r14d
  char v15; // di
  __int64 v16; // rdx
  __int64 v17; // rsi
  int v18; // r9d
  unsigned int v19; // r8d
  __int64 v20; // rdx
  _QWORD *v21; // rcx
  __int64 v22; // rcx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 *v27; // r15
  __int64 v28; // rdi
  int v29; // eax
  int v30; // edx
  int v31; // r10d
  int v32; // r10d
  __int64 *v33; // [rsp+8h] [rbp-78h]
  _QWORD *v34; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v35; // [rsp+18h] [rbp-68h] BYREF
  char v36[96]; // [rsp+20h] [rbp-60h] BYREF

  v6 = *(_BYTE *)(a2 + 8) & 1;
  if ( v6 )
  {
    v7 = a2 + 16;
    v8 = 3;
  }
  else
  {
    v24 = *(unsigned int *)(a2 + 24);
    v7 = *(_QWORD *)(a2 + 16);
    if ( !(_DWORD)v24 )
      goto LABEL_20;
    v8 = v24 - 1;
  }
  v9 = *a1;
  v10 = v8 & (((unsigned int)*a1 >> 9) ^ ((unsigned int)*a1 >> 4));
  v11 = (__int64 *)(v7 + 16LL * (v8 & (((unsigned int)*a1 >> 9) ^ ((unsigned int)*a1 >> 4))));
  v12 = *v11;
  if ( *a1 == *v11 )
    goto LABEL_4;
  v29 = 1;
  while ( v12 != -8 )
  {
    v32 = v29 + 1;
    v10 = v8 & (v29 + v10);
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( v9 == *v11 )
      goto LABEL_4;
    v29 = v32;
  }
  if ( v6 )
  {
    v25 = 64;
    goto LABEL_21;
  }
  v24 = *(unsigned int *)(a2 + 24);
LABEL_20:
  v25 = 16 * v24;
LABEL_21:
  v11 = (__int64 *)(v7 + v25);
LABEL_4:
  v13 = 64;
  if ( !v6 )
    v13 = 16LL * *(unsigned int *)(a2 + 24);
  v14 = 0;
  if ( v11 != (__int64 *)(v13 + v7) )
  {
    v15 = *(_BYTE *)(a3 + 8) & 1;
    if ( v15 )
    {
      v17 = a3 + 16;
      v18 = 3;
    }
    else
    {
      v16 = *(unsigned int *)(a3 + 24);
      v17 = *(_QWORD *)(a3 + 16);
      if ( !(_DWORD)v16 )
        goto LABEL_23;
      v18 = v16 - 1;
    }
    v19 = v18 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v20 = v17 + 16LL * v19;
    v21 = *(_QWORD **)v20;
    if ( a1 == *(_QWORD **)v20 )
    {
LABEL_11:
      v22 = 64;
      if ( !v15 )
        v22 = 16LL * *(unsigned int *)(a3 + 24);
      if ( v20 == v17 + v22 )
      {
        v14 = *((_DWORD *)v11 + 2);
        v33 = (__int64 *)a1[4];
        if ( (__int64 *)a1[3] != v33 )
        {
          v27 = (__int64 *)a1[3];
          do
          {
            v28 = *v27++;
            v14 += sub_1A55C50(v28, a2, a3);
          }
          while ( v33 != v27 );
        }
        v34 = a1;
        v35 = v14;
        sub_1A55A00((__int64)v36, a3, (__int64 *)&v34, &v35);
      }
      else
      {
        return *(unsigned int *)(v20 + 8);
      }
      return v14;
    }
    v30 = 1;
    while ( v21 != (_QWORD *)-8LL )
    {
      v31 = v30 + 1;
      v19 = v18 & (v30 + v19);
      v20 = v17 + 16LL * v19;
      v21 = *(_QWORD **)v20;
      if ( a1 == *(_QWORD **)v20 )
        goto LABEL_11;
      v30 = v31;
    }
    if ( v15 )
    {
      v26 = 64;
      goto LABEL_24;
    }
    v16 = *(unsigned int *)(a3 + 24);
LABEL_23:
    v26 = 16 * v16;
LABEL_24:
    v20 = v17 + v26;
    goto LABEL_11;
  }
  return v14;
}
