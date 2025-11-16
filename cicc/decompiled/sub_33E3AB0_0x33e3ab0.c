// Function: sub_33E3AB0
// Address: 0x33e3ab0
//
void __fastcall sub_33E3AB0(char a1, unsigned int a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6, _QWORD *a7)
{
  int v7; // r12d
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 *v12; // r14
  unsigned int v13; // r15d
  int v14; // r13d
  unsigned int v15; // ecx
  unsigned int v16; // ebx
  unsigned int v17; // r12d
  int v18; // eax
  __int64 v19; // r15
  char v20; // r10
  __int64 *v21; // rdi
  __int64 v22; // r9
  __int64 v23; // rsi
  unsigned int v24; // ecx
  unsigned int i; // eax
  int v26; // eax
  unsigned int v27; // r14d
  _QWORD *v28; // r10
  int v29; // r13d
  _QWORD *v30; // rdx
  unsigned int v31; // r12d
  int v32; // ebx
  unsigned int v33; // r13d
  int v34; // r14d
  _QWORD *v35; // r15
  int v36; // ecx
  unsigned int v37; // ecx
  unsigned int v38; // [rsp+8h] [rbp-88h]
  int v39; // [rsp+Ch] [rbp-84h]
  unsigned int v40; // [rsp+Ch] [rbp-84h]
  __int64 *v41; // [rsp+10h] [rbp-80h]
  unsigned int v42; // [rsp+10h] [rbp-80h]
  unsigned int v43; // [rsp+18h] [rbp-78h]
  __int64 v44; // [rsp+18h] [rbp-78h]
  unsigned int v47; // [rsp+40h] [rbp-50h]
  int v48; // [rsp+40h] [rbp-50h]
  __int64 v49; // [rsp+40h] [rbp-50h]
  unsigned int v50; // [rsp+48h] [rbp-48h]
  int v51; // [rsp+48h] [rbp-48h]
  int v52; // [rsp+48h] [rbp-48h]
  unsigned __int64 v54; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v55; // [rsp+58h] [rbp-38h]

  v7 = a5;
  v8 = *(_DWORD *)(a4 + 8);
  v41 = (__int64 *)a4;
  v39 = a5;
  *(_DWORD *)(a6 + 8) = 0;
  v50 = a5 * v8;
  v38 = (unsigned int)a5 * v8 / a2;
  LOBYTE(a4) = v38;
  *(_DWORD *)(a6 + 64) = v38;
  LODWORD(v9) = (v38 + 63) >> 6;
  if ( (_DWORD)v9 )
  {
    v10 = 0;
    v9 = (unsigned int)v9;
    if ( *(_DWORD *)(a6 + 12) < (unsigned int)v9 )
    {
      v49 = (unsigned int)v9;
      sub_C8D5F0(a6, (const void *)(a6 + 16), (unsigned int)v9, 8u, a5, a6);
      v9 = v49;
      v10 = 8LL * *(unsigned int *)(a6 + 8);
    }
    memset((void *)(*(_QWORD *)a6 + v10), 0, 8 * v9);
    *(_DWORD *)(a6 + 8) += (v38 + 63) >> 6;
    LODWORD(a4) = *(_DWORD *)(a6 + 64);
  }
  v11 = a4 & 0x3F;
  if ( (_DWORD)v11 )
    *(_QWORD *)(*(_QWORD *)a6 + 8LL * *(unsigned int *)(a6 + 8) - 8) &= ~(-1LL << v11);
  v55 = a2;
  if ( a2 > 0x40 )
    sub_C43690((__int64)&v54, 0, 0);
  else
    v54 = 0;
  sub_33E3890((__int64)a3, v38, (__int64)&v54, v11, a5, a6);
  if ( v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  if ( a2 >= v8 )
  {
    v40 = 0;
    v26 = a2 / v8;
    if ( v50 >= a2 )
    {
      v27 = v8;
      v28 = a7;
      v29 = 0;
      do
      {
        v52 = v26 - 1;
        v30 = (_QWORD *)(8LL * (v40 >> 6) + *(_QWORD *)a6);
        *v30 |= 1LL << v40;
        v44 = *a3 + 16LL * v40;
        v31 = 0;
        v32 = v29;
        v33 = v27;
        v34 = 0;
        v35 = v28;
        do
        {
          v36 = v52 - v34;
          if ( a1 )
            v36 = v34;
          v37 = v32 + v36;
          if ( (*(_QWORD *)(*v35 + 8LL * (v37 >> 6)) & (1LL << v37)) == 0 )
          {
            v48 = v26;
            *(_QWORD *)(*(_QWORD *)a6 + 8LL * (v40 >> 6)) &= ~(1LL << v40);
            sub_C43D80(v44, (__int64)&v41[2 * v37], v31);
            v26 = v48;
          }
          ++v34;
          v31 += v33;
        }
        while ( v26 != v34 );
        v27 = v33;
        ++v40;
        v28 = v35;
        v29 = v26 + v32;
      }
      while ( v38 != v40 );
    }
  }
  else
  {
    v51 = v8 / a2;
    if ( v7 )
    {
      v12 = v41;
      v47 = 0;
      v42 = v8 / a2;
      v43 = 0;
      v13 = a2;
      do
      {
        v14 = 0;
        v15 = v43++;
        v16 = 0;
        v17 = v13;
        if ( (*(_QWORD *)(*a7 + 8LL * (v15 >> 6)) & (1LL << v15)) != 0 )
        {
          if ( v47 != v42 )
          {
            v20 = v47 & 0x3F;
            v21 = (__int64 *)(*(_QWORD *)a6 + 8LL * (v47 >> 6));
            v22 = *v21;
            v23 = 1LL << v42;
            if ( v47 >> 6 == v42 >> 6 )
            {
              *v21 = v22 | (v23 - (1LL << v20));
            }
            else
            {
              *v21 = v22 | (-1LL << v20);
              v24 = ((v47 != 0) + (unsigned int)((v47 - (unsigned __int64)(v47 != 0)) >> 6)) << 6;
              for ( i = v24 + 64; v42 >= i; i += 64 )
              {
                *(_QWORD *)(*(_QWORD *)a6 + 8LL * ((i - 64) >> 6)) = -1;
                v24 = i;
              }
              if ( v24 < v42 )
                *(_QWORD *)(*(_QWORD *)a6 + 8LL * (v24 >> 6)) |= v23 - 1;
            }
          }
        }
        else
        {
          do
          {
            v18 = v51 - 1 - v14;
            if ( a1 )
              v18 = v14;
            v19 = *a3 + 16LL * (v47 + v18);
            sub_C440A0((__int64)&v54, v12, v17, v16);
            if ( *(_DWORD *)(v19 + 8) > 0x40u )
            {
              if ( *(_QWORD *)v19 )
                j_j___libc_free_0_0(*(_QWORD *)v19);
            }
            ++v14;
            v16 += v17;
            *(_QWORD *)v19 = v54;
            *(_DWORD *)(v19 + 8) = v55;
          }
          while ( v51 != v14 );
          v13 = v17;
        }
        v12 += 2;
        v42 += v51;
        v47 += v51;
      }
      while ( v39 != v43 );
    }
  }
}
