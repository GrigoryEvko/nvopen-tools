// Function: sub_2523990
// Address: 0x2523990
//
void __fastcall sub_2523990(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 *v8; // r12
  __int64 *v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 *v13; // r10
  __int64 v14; // rax
  __int64 *v15; // rbx
  __int64 *v16; // r14
  __int64 v17; // rcx
  char v18; // al
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 *v24; // rax
  __int64 *v25; // r12
  __int64 *v26; // rbx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rsi
  int v32; // r10d
  unsigned int i; // eax
  unsigned int v34; // eax
  __int64 *v35; // [rsp-10h] [rbp-120h]
  __int64 v36; // [rsp-8h] [rbp-118h]
  char v37; // [rsp+8h] [rbp-108h]
  char v38; // [rsp+1Bh] [rbp-F5h] BYREF
  unsigned int v39; // [rsp+1Ch] [rbp-F4h] BYREF
  __int64 v40[2]; // [rsp+20h] [rbp-F0h] BYREF
  __int64 *v41; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v42; // [rsp+38h] [rbp-D8h]
  _BYTE v43[64]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v44; // [rsp+80h] [rbp-90h] BYREF
  __int64 *v45; // [rsp+88h] [rbp-88h]
  __int64 v46; // [rsp+90h] [rbp-80h]
  int v47; // [rsp+98h] [rbp-78h]
  char v48; // [rsp+9Ch] [rbp-74h]
  char v49; // [rsp+A0h] [rbp-70h] BYREF

  v6 = 0;
  v7 = *(_QWORD *)(a1 + 200);
  if ( !*(_BYTE *)(a1 + 4296) )
  {
    v27 = *(unsigned int *)(v7 + 40);
    a5 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 8 * v27 - 8);
    v28 = *(_QWORD *)(*(_QWORD *)(a1 + 208) + 240LL);
    v6 = *(_QWORD *)v28;
    if ( *(_QWORD *)v28 )
    {
      if ( *(_BYTE *)(v28 + 16) )
      {
        v30 = *(unsigned int *)(v6 + 88);
        v31 = *(_QWORD *)(v6 + 72);
        if ( !(_DWORD)v30 )
          goto LABEL_47;
        v32 = 1;
        for ( i = (v30 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4)
                    | ((unsigned __int64)(((unsigned int)&unk_4F6D3F8 >> 9) ^ ((unsigned int)&unk_4F6D3F8 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4)))); ; i = (v30 - 1) & v34 )
        {
          a6 = v31 + 24LL * i;
          if ( *(_UNKNOWN **)a6 == &unk_4F6D3F8 && a5 == *(_QWORD *)(a6 + 8) )
            break;
          if ( *(_QWORD *)a6 == -4096 && *(_QWORD *)(a6 + 8) == -4096 )
            goto LABEL_47;
          v34 = v32 + i;
          ++v32;
        }
        if ( a6 == v31 + 24 * v30 )
        {
LABEL_47:
          v6 = 0;
        }
        else
        {
          v6 = *(_QWORD *)(*(_QWORD *)(a6 + 16) + 24LL);
          if ( v6 )
            v6 += 8;
        }
      }
      else
      {
        v29 = sub_BC1CD0(*(_QWORD *)v28, &unk_4F6D3F8, *(_QWORD *)(*(_QWORD *)(v7 + 32) + 8 * v27 - 8));
        v7 = *(_QWORD *)(a1 + 200);
        v6 = v29 + 8;
      }
    }
  }
  v8 = *(__int64 **)(v7 + 32);
  v41 = (__int64 *)v43;
  v42 = 0x800000000LL;
  v9 = &v8[*(unsigned int *)(v7 + 40)];
  if ( v8 == v9 )
  {
    v13 = (__int64 *)v43;
    v14 = 0;
  }
  else
  {
    do
    {
      v10 = *v8;
      if ( (*(_BYTE *)(*v8 + 32) & 0xFu) - 7 <= 1 && (*(_BYTE *)(a1 + 4296) || !sub_981210(*(_QWORD *)v6, *v8, &v39)) )
      {
        v11 = (unsigned int)v42;
        v12 = (unsigned int)v42 + 1LL;
        if ( v12 > HIDWORD(v42) )
        {
          sub_C8D5F0((__int64)&v41, v43, v12, 8u, a5, a6);
          v11 = (unsigned int)v42;
        }
        v41[v11] = v10;
        LODWORD(v42) = v42 + 1;
      }
      ++v8;
    }
    while ( v9 != v8 );
    v13 = v41;
    v14 = (unsigned int)v42;
  }
  v46 = 8;
  v45 = (__int64 *)&v49;
  v44 = 0;
  v47 = 0;
  v48 = 1;
  do
  {
    v15 = &v13[v14];
    if ( v15 == v13 )
      goto LABEL_29;
    v37 = 0;
    v16 = v13;
    do
    {
      if ( *v16 )
      {
        v38 = 0;
        v17 = *v16;
        v40[0] = a1;
        v40[1] = (__int64)&v44;
        v18 = sub_25230B0(a1, (__int64 (__fastcall *)(__int64, __int64 *))sub_2508250, (__int64)v40, v17, 1, 0, &v38, 0);
        v21 = v35;
        v22 = v36;
        if ( !v18 )
        {
          v23 = *v16;
          if ( !v48 )
            goto LABEL_34;
          v24 = v45;
          v22 = HIDWORD(v46);
          v21 = &v45[HIDWORD(v46)];
          if ( v45 != v21 )
          {
            while ( v23 != *v24 )
            {
              if ( v21 == ++v24 )
                goto LABEL_35;
            }
            goto LABEL_21;
          }
LABEL_35:
          if ( HIDWORD(v46) < (unsigned int)v46 )
          {
            ++HIDWORD(v46);
            *v21 = v23;
            ++v44;
          }
          else
          {
LABEL_34:
            sub_C8CC70((__int64)&v44, v23, (__int64)v21, v22, v19, v20);
          }
LABEL_21:
          *v16 = 0;
          v37 = 1;
        }
      }
      ++v16;
    }
    while ( v15 != v16 );
    v13 = v41;
    v14 = (unsigned int)v42;
  }
  while ( v37 );
  v25 = &v41[(unsigned int)v42];
  if ( v41 != v25 )
  {
    v26 = v41;
    do
    {
      v40[0] = *v26;
      if ( v40[0] )
        sub_2518560(a1 + 3656, v40);
      ++v26;
    }
    while ( v25 != v26 );
  }
LABEL_29:
  if ( !v48 )
    _libc_free((unsigned __int64)v45);
  if ( v41 != (__int64 *)v43 )
    _libc_free((unsigned __int64)v41);
}
