// Function: sub_B00540
// Address: 0xb00540
//
__int64 __fastcall sub_B00540(unsigned __int8 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // r13
  bool v6; // zf
  unsigned __int8 *v7; // r12
  __int64 i; // rcx
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int8 **v11; // rdx
  unsigned __int8 *v12; // r9
  int v13; // edx
  int v14; // r10d
  __int64 v15; // r12
  _BYTE *v16; // r15
  __int64 v17; // r10
  __int64 v18; // rcx
  _QWORD *v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // r9
  _QWORD *v22; // rax
  __int64 v23; // r8
  __int64 *v24; // rax
  __int64 v25; // r13
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rax
  int v29; // esi
  int v30; // eax
  int v31; // eax
  int v32; // esi
  _QWORD *v33; // r10
  _BYTE *v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+28h] [rbp-78h] BYREF
  _BYTE *v37; // [rsp+30h] [rbp-70h] BYREF
  __int64 v38; // [rsp+38h] [rbp-68h]
  _BYTE v39[96]; // [rsp+40h] [rbp-60h] BYREF

  v4 = a2;
  v6 = *a1 == 18;
  v37 = v39;
  v38 = 0x600000000LL;
  if ( !v6 )
  {
    v7 = a1;
    for ( i = 0; ; i = (unsigned int)v38 )
    {
      v9 = *(unsigned int *)(a4 + 24);
      v10 = *(_QWORD *)(a4 + 8);
      if ( (_DWORD)v9 )
      {
        a2 = ((_DWORD)v9 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v11 = (unsigned __int8 **)(v10 + 16 * a2);
        v12 = *v11;
        if ( v7 == *v11 )
        {
LABEL_7:
          if ( v11 != (unsigned __int8 **)(v10 + 16 * v9) )
          {
            v15 = (__int64)v11[1];
            v35 = v37;
            v16 = &v37[8 * i];
            if ( !v15 )
              goto LABEL_50;
            goto LABEL_15;
          }
        }
        else
        {
          v13 = 1;
          while ( v12 != (unsigned __int8 *)-4096LL )
          {
            v14 = v13 + 1;
            a2 = ((_DWORD)v9 - 1) & (unsigned int)(v13 + a2);
            v11 = (unsigned __int8 **)(v10 + 16LL * (unsigned int)a2);
            v12 = *v11;
            if ( *v11 == v7 )
              goto LABEL_7;
            v13 = v14;
          }
        }
      }
      if ( i + 1 > (unsigned __int64)HIDWORD(v38) )
      {
        a2 = (unsigned __int64)v39;
        sub_C8D5F0(&v37, v39, i + 1, 8);
        i = (unsigned int)v38;
      }
      *(_QWORD *)&v37[8 * i] = v7;
      LODWORD(v38) = v38 + 1;
      v7 = (unsigned __int8 *)sub_AF2660(v7);
      if ( *v7 == 18 )
      {
        v35 = v37;
        v16 = &v37[8 * (unsigned int)v38];
        goto LABEL_50;
      }
    }
  }
  v35 = v39;
  v16 = v39;
LABEL_50:
  v15 = v4;
LABEL_15:
  if ( v16 == v35 )
    goto LABEL_46;
  do
  {
    v25 = *((_QWORD *)v16 - 1);
    sub_B9C990(&v36, v25);
    sub_B97110(v36, 1, v15);
    v26 = v36;
    v36 = 0;
    v28 = sub_BA6670(v26, 1, v27);
    a2 = *(unsigned int *)(a4 + 24);
    v15 = v28;
    if ( (_DWORD)a2 )
    {
      v17 = *(_QWORD *)(a4 + 8);
      v18 = 1;
      v19 = 0;
      v20 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
      v21 = ((_DWORD)a2 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v22 = (_QWORD *)(v17 + 16 * v21);
      v23 = *v22;
      if ( v25 == *v22 )
      {
LABEL_18:
        v24 = v22 + 1;
        goto LABEL_19;
      }
      while ( v23 != -4096 )
      {
        if ( !v19 && v23 == -8192 )
          v19 = v22;
        v21 = ((_DWORD)a2 - 1) & (unsigned int)(v18 + v21);
        v22 = (_QWORD *)(v17 + 16LL * (unsigned int)v21);
        v23 = *v22;
        if ( v25 == *v22 )
          goto LABEL_18;
        v18 = (unsigned int)(v18 + 1);
      }
      if ( !v19 )
        v19 = v22;
      v31 = *(_DWORD *)(a4 + 16);
      ++*(_QWORD *)a4;
      v30 = v31 + 1;
      if ( 4 * v30 < (unsigned int)(3 * a2) )
      {
        v23 = (unsigned int)(a2 - *(_DWORD *)(a4 + 20) - v30);
        v21 = (unsigned int)a2 >> 3;
        if ( (unsigned int)v23 > (unsigned int)v21 )
          goto LABEL_26;
        sub_B00360(a4, a2);
        v32 = *(_DWORD *)(a4 + 24);
        if ( !v32 )
        {
LABEL_64:
          ++*(_DWORD *)(a4 + 16);
          BUG();
        }
        a2 = (unsigned int)(v32 - 1);
        v21 = *(_QWORD *)(a4 + 8);
        v33 = 0;
        v18 = 1;
        v20 = (unsigned int)a2 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v30 = *(_DWORD *)(a4 + 16) + 1;
        v19 = (_QWORD *)(v21 + 16 * v20);
        v23 = *v19;
        if ( v25 == *v19 )
          goto LABEL_26;
        while ( v23 != -4096 )
        {
          if ( !v33 && v23 == -8192 )
            v33 = v19;
          v20 = (unsigned int)a2 & ((_DWORD)v18 + (_DWORD)v20);
          v19 = (_QWORD *)(v21 + 16LL * (unsigned int)v20);
          v23 = *v19;
          if ( v25 == *v19 )
            goto LABEL_26;
          v18 = (unsigned int)(v18 + 1);
        }
        goto LABEL_42;
      }
    }
    else
    {
      ++*(_QWORD *)a4;
    }
    sub_B00360(a4, 2 * a2);
    v29 = *(_DWORD *)(a4 + 24);
    if ( !v29 )
      goto LABEL_64;
    a2 = (unsigned int)(v29 - 1);
    v21 = *(_QWORD *)(a4 + 8);
    v20 = (unsigned int)a2 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
    v30 = *(_DWORD *)(a4 + 16) + 1;
    v19 = (_QWORD *)(v21 + 16 * v20);
    v23 = *v19;
    if ( v25 == *v19 )
      goto LABEL_26;
    v18 = 1;
    v33 = 0;
    while ( v23 != -4096 )
    {
      if ( !v33 && v23 == -8192 )
        v33 = v19;
      v20 = (unsigned int)a2 & ((_DWORD)v18 + (_DWORD)v20);
      v19 = (_QWORD *)(v21 + 16LL * (unsigned int)v20);
      v23 = *v19;
      if ( v25 == *v19 )
        goto LABEL_26;
      v18 = (unsigned int)(v18 + 1);
    }
LABEL_42:
    if ( v33 )
      v19 = v33;
LABEL_26:
    *(_DWORD *)(a4 + 16) = v30;
    if ( *v19 != -4096 )
      --*(_DWORD *)(a4 + 20);
    *v19 = v25;
    v24 = v19 + 1;
    v19[1] = 0;
LABEL_19:
    *v24 = v15;
    if ( v36 )
      sub_BA65D0(v36, a2, v20, v18, v23, v21);
    v16 -= 8;
  }
  while ( v16 != v35 );
  v35 = v37;
LABEL_46:
  if ( v35 != v39 )
    _libc_free(v35, a2);
  return v15;
}
