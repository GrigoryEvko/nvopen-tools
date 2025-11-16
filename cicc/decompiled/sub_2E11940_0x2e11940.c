// Function: sub_2E11940
// Address: 0x2e11940
//
void __fastcall sub_2E11940(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *i; // rdx
  __int64 v12; // rdi
  __int64 j; // rax
  unsigned int *v14; // r14
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // r8
  unsigned __int64 v18; // r9
  __int64 v19; // r14
  __int64 v20; // rcx
  unsigned int v21; // ebx
  __int64 v22; // rdx
  __int64 v23; // r13
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int *v29; // rdi
  unsigned int *v30; // r12
  unsigned int *v31; // rbx
  __int64 v32; // rcx
  __int64 v33; // [rsp+10h] [rbp-90h]
  __int64 v34; // [rsp+18h] [rbp-88h]
  char v35; // [rsp+28h] [rbp-78h]
  unsigned int *v36; // [rsp+30h] [rbp-70h]
  unsigned int *v37; // [rsp+38h] [rbp-68h]
  unsigned int *v38; // [rsp+40h] [rbp-60h] BYREF
  __int64 v39; // [rsp+48h] [rbp-58h]
  _BYTE v40[80]; // [rsp+50h] [rbp-50h] BYREF

  v7 = *(unsigned int *)(a1[2] + 44LL);
  v8 = *((unsigned int *)a1 + 108);
  if ( v7 != v8 )
  {
    if ( v7 >= v8 )
    {
      if ( v7 > *((unsigned int *)a1 + 109) )
      {
        sub_C8D5F0((__int64)(a1 + 53), a1 + 55, v7, 8u, a5, a6);
        v8 = *((unsigned int *)a1 + 108);
      }
      v9 = a1[53];
      v10 = (_QWORD *)(v9 + 8 * v8);
      for ( i = (_QWORD *)(v9 + 8 * v7); i != v10; ++v10 )
      {
        if ( v10 )
          *v10 = 0;
      }
    }
    *((_DWORD *)a1 + 108) = v7;
  }
  v38 = (unsigned int *)v40;
  v39 = 0x800000000LL;
  v12 = *(_QWORD *)(*a1 + 328LL);
  v33 = *a1 + 320LL;
  v34 = v12;
  if ( v12 != v33 )
  {
    for ( j = v12; ; j = *(_QWORD *)(*a1 + 328LL) )
    {
      if ( v34 == j || *(_BYTE *)(v34 + 216) )
      {
        v14 = *(unsigned int **)(v34 + 192);
        v36 = v14;
        if ( *(unsigned int **)(v34 + 184) != v14 )
        {
          v15 = *(_QWORD *)(*(_QWORD *)(a1[4] + 152LL) + 16LL * *(unsigned int *)(v34 + 24));
          v16 = sub_2E33140();
          if ( v14 != (unsigned int *)v16 )
          {
            v37 = (unsigned int *)v16;
            do
            {
              v22 = a1[2];
              v20 = *(_QWORD *)(v22 + 8);
              v21 = *(_DWORD *)(v20 + 24LL * *v37 + 16) & 0xFFF;
              v19 = *(_QWORD *)(v22 + 56) + 2LL * (*(_DWORD *)(v20 + 24LL * *v37 + 16) >> 12);
              do
              {
                if ( !v19 )
                  break;
                v23 = *(_QWORD *)(a1[53] + 8LL * v21);
                if ( !v23 )
                {
                  v35 = qword_501EA48[8];
                  v24 = (_QWORD *)sub_22077B0(0x68u);
                  v25 = v21;
                  v23 = (__int64)v24;
                  if ( v24 )
                  {
                    v24[1] = 0x200000000LL;
                    v24[9] = 0x200000000LL;
                    *v24 = v24 + 2;
                    v24[8] = v24 + 10;
                    if ( v35 )
                    {
                      v28 = sub_22077B0(0x30u);
                      v25 = v21;
                      if ( v28 )
                      {
                        *(_DWORD *)(v28 + 8) = 0;
                        *(_QWORD *)(v28 + 16) = 0;
                        *(_QWORD *)(v28 + 24) = v28 + 8;
                        *(_QWORD *)(v28 + 32) = v28 + 8;
                        *(_QWORD *)(v28 + 40) = 0;
                      }
                      *(_QWORD *)(v23 + 96) = v28;
                    }
                    else
                    {
                      v24[12] = 0;
                    }
                  }
                  *(_QWORD *)(a1[53] + 8 * v25) = v23;
                  v26 = (unsigned int)v39;
                  v20 = HIDWORD(v39);
                  v27 = (unsigned int)v39 + 1LL;
                  if ( v27 > HIDWORD(v39) )
                  {
                    sub_C8D5F0((__int64)&v38, v40, v27, 4u, v17, v18);
                    v26 = (unsigned int)v39;
                  }
                  v38[v26] = v21;
                  LODWORD(v39) = v39 + 1;
                }
                v19 += 2;
                sub_2E0E0B0(v23, v15, a1 + 7, v20, v17, v18);
                v21 += *(__int16 *)(v19 - 2);
              }
              while ( *(_WORD *)(v19 - 2) );
              v37 += 6;
            }
            while ( v36 != v37 );
          }
        }
      }
      v34 = *(_QWORD *)(v34 + 8);
      if ( v33 == v34 )
        break;
    }
    v29 = v38;
    v30 = &v38[(unsigned int)v39];
    if ( v30 != v38 )
    {
      v31 = v38;
      do
      {
        v32 = *v31++;
        sub_2E11710(a1, *(_QWORD *)(a1[53] + 8 * v32), v32);
      }
      while ( v30 != v31 );
      v29 = v38;
    }
    if ( v29 != (unsigned int *)v40 )
      _libc_free((unsigned __int64)v29);
  }
}
