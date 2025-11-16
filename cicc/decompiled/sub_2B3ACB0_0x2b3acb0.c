// Function: sub_2B3ACB0
// Address: 0x2b3acb0
//
__int64 __fastcall sub_2B3ACB0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rbx
  unsigned int v7; // eax
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v11; // eax
  unsigned __int64 v12; // r8
  char *v13; // rdx
  __int64 v14; // rax
  char *v15; // rdi
  unsigned int v16; // r13d
  unsigned __int64 v17; // r12
  unsigned __int64 v19; // r13
  unsigned __int64 v20; // rbx
  char *v21; // r13
  char *v22; // rsi
  char *v23; // rax
  int v24; // edx
  char v25; // dl
  __int64 v26; // rax
  unsigned __int64 v27; // r8
  __int64 v28; // rcx
  unsigned __int64 v29; // rax
  char *v30; // rdi
  char *v31; // r11
  int v32; // edx
  unsigned int *v33; // rax
  unsigned int v34; // edi
  unsigned int v35; // edx
  _DWORD *v36; // rsi
  unsigned __int64 v37; // [rsp-8h] [rbp-B8h]
  __int64 v40; // [rsp+20h] [rbp-90h]
  __int64 v41; // [rsp+28h] [rbp-88h]
  char *v42; // [rsp+28h] [rbp-88h]
  char *v43; // [rsp+40h] [rbp-70h] BYREF
  __int64 v44; // [rsp+48h] [rbp-68h]
  __int64 v45; // [rsp+50h] [rbp-60h] BYREF
  char v46[88]; // [rsp+58h] [rbp-58h] BYREF

  v43 = (char *)&v45;
  v4 = *a2;
  v44 = 0x600000001LL;
  v45 = 0;
  v5 = *(_QWORD *)(v4 - 32);
  v40 = (unsigned int)a3;
  v41 = *(_QWORD *)(*(_QWORD *)(v4 - 64) + 8LL);
  if ( (_DWORD)a3 == 1 )
  {
    v16 = 0;
    if ( a3 != 1 )
      return v16;
    v21 = v46;
    v17 = 1;
    sub_2B0DEF0(&v45, v46, 0);
    v30 = (char *)&v45;
    goto LABEL_35;
  }
  v6 = 1;
  do
  {
    v7 = sub_D35010(
           v41,
           v5,
           *(_QWORD *)(*(_QWORD *)(a2[(unsigned int)v6] - 64) + 8LL),
           *(_QWORD *)(a2[(unsigned int)v6] - 32),
           *(_QWORD *)(a1 + 3344),
           *(_QWORD *)(a1 + 3288),
           1,
           1);
    v9 = (unsigned int)v44;
    v10 = v7;
    v11 = v44;
    v12 = v37;
    if ( (unsigned int)v44 >= (unsigned __int64)HIDWORD(v44) )
    {
      v12 = (unsigned int)v44 + 1LL;
      v19 = v10 | ((unsigned __int64)(unsigned int)v6 << 32);
      if ( HIDWORD(v44) < v12 )
      {
        sub_C8D5F0((__int64)&v43, &v45, (unsigned int)v44 + 1LL, 8u, v12, v8);
        v9 = (unsigned int)v44;
      }
      *(_QWORD *)&v43[8 * v9] = v19;
      LODWORD(v44) = v44 + 1;
    }
    else
    {
      v13 = &v43[8 * (unsigned int)v44];
      if ( v13 )
      {
        *(_DWORD *)v13 = v10;
        *((_DWORD *)v13 + 1) = v6;
        v11 = v44;
      }
      LODWORD(v44) = v11 + 1;
    }
    ++v6;
  }
  while ( v40 != v6 );
  v14 = (unsigned int)v44;
  v15 = v43;
  v16 = 0;
  v17 = (unsigned int)v44;
  if ( a3 == (unsigned int)v44 )
  {
    v20 = 8LL * (unsigned int)v44;
    v21 = &v43[v20];
    if ( &v43[v20] == v43 )
    {
LABEL_16:
      v22 = &v21[8 * v14];
      if ( v22 != v21 )
      {
        v23 = v21;
        v24 = 0;
        while ( 1 )
        {
          v23 += 8;
          ++v24;
          if ( v23 == v22 )
            break;
          if ( v24 )
          {
            v10 = (unsigned int)(*((_DWORD *)v23 - 2) + 1);
            if ( *(_DWORD *)v23 != (_DWORD)v10 )
            {
              v15 = v21;
              v16 = 0;
              goto LABEL_9;
            }
          }
        }
      }
      sub_2B39CB0(a4, v17, 0, v10, v12, v8);
      v15 = v43;
      if ( &v43[8 * (unsigned int)v44] == v43 )
        goto LABEL_26;
      v25 = 1;
      v26 = 0;
      v27 = (8 * (unsigned __int64)(unsigned int)v44 - 8) >> 3;
      do
      {
        *(_DWORD *)(*(_QWORD *)a4 + 4LL * *(unsigned int *)&v15[8 * v26 + 4]) = v26;
        v25 &= *(_DWORD *)&v15[8 * v26 + 4] == v26;
        v28 = v26++;
      }
      while ( v28 != v27 );
      v15 = v43;
      v16 = 1;
      if ( v25 )
      {
LABEL_26:
        v16 = 1;
        *(_DWORD *)(a4 + 8) = 0;
      }
      goto LABEL_9;
    }
    v42 = v43;
    _BitScanReverse64(&v29, (__int64)v20 >> 3);
    sub_2B0DEF0(v43, &v43[8 * (unsigned int)v44], 2LL * (int)(63 - (v29 ^ 0x3F)));
    v30 = v42;
    if ( v20 > 0x80 )
    {
      sub_2B0A810(v42, v42 + 128);
      for ( ; v21 != v31; v36[1] = v34 )
      {
        v10 = *(unsigned int *)v31;
        v32 = *((_DWORD *)v31 - 2);
        v33 = (unsigned int *)(v31 - 8);
        v34 = *((_DWORD *)v31 + 1);
        if ( v32 <= (int)v10 )
        {
          v36 = v31;
        }
        else
        {
          do
          {
            v33[2] = v32;
            v35 = v33[1];
            v36 = v33;
            v33 -= 2;
            v33[5] = v35;
            v32 = *v33;
          }
          while ( (int)v10 < (int)*v33 );
        }
        v31 += 8;
        *v36 = v10;
      }
      goto LABEL_32;
    }
LABEL_35:
    sub_2B0A810(v30, v21);
LABEL_32:
    v21 = v43;
    v14 = (unsigned int)v44;
    goto LABEL_16;
  }
LABEL_9:
  if ( v15 != (char *)&v45 )
    _libc_free((unsigned __int64)v15);
  return v16;
}
