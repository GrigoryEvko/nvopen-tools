// Function: sub_1DB5D80
// Address: 0x1db5d80
//
__int64 __fastcall sub_1DB5D80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r15
  __int64 v5; // r13
  int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // r14
  _QWORD *v9; // rax
  int v10; // r8d
  __int64 *v11; // r9
  _QWORD *v12; // r12
  int v13; // r13d
  __int64 *v14; // rbx
  __int64 v15; // r15
  __int64 v16; // rax
  int v17; // r9d
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // r12
  __m128i *v21; // r9
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __m128i *v28; // rax
  __int64 v29; // rax
  __int64 result; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  bool v33; // zf
  __int64 v34; // [rsp+0h] [rbp-A0h]
  __int64 v35; // [rsp+8h] [rbp-98h]
  const void *v37; // [rsp+18h] [rbp-88h]
  const void *v38; // [rsp+20h] [rbp-80h]
  unsigned int v39; // [rsp+2Ch] [rbp-74h]
  unsigned int v41; // [rsp+38h] [rbp-68h]
  int v42; // [rsp+3Ch] [rbp-64h]
  __int64 *v43; // [rsp+40h] [rbp-60h]
  __int64 *v44; // [rsp+48h] [rbp-58h]
  __m128i v45; // [rsp+50h] [rbp-50h] BYREF
  __int64 v46; // [rsp+60h] [rbp-40h]

  v4 = (__int64 *)a2;
  v5 = *(_QWORD *)(a1 + 104);
  v41 = a3;
  if ( v5 )
  {
    v39 = a3;
    v44 = (__int64 *)a2;
    do
    {
      v6 = *(_DWORD *)(v5 + 112);
      v7 = v6 & v41;
      v42 = v6 & v41;
      if ( (v6 & v41) != 0 )
      {
        v8 = v5;
        if ( (_DWORD)v7 != v6 )
        {
          a2 = (__int64)v44;
          v7 = 120;
          *(_DWORD *)(v5 + 112) = ~(v6 & v41) & v6;
          v9 = (_QWORD *)sub_145CDC0(0x78u, v44);
          v8 = (__int64)v9;
          if ( v9 )
          {
            a2 = (__int64)(v9 + 10);
            v9[12] = 0;
            v38 = v9 + 2;
            *v9 = v9 + 2;
            v9[1] = 0x200000000LL;
            v37 = v9 + 10;
            v9[8] = v9 + 10;
            v9[9] = 0x200000000LL;
            if ( (_QWORD *)v5 != v9 )
            {
              v11 = *(__int64 **)(v5 + 64);
              v35 = (__int64)(v9 + 8);
              v43 = &v11[*(unsigned int *)(v5 + 72)];
              if ( v11 != v43 )
              {
                v34 = v5;
                v12 = v9;
                v13 = 0;
                v14 = v11;
                do
                {
                  a2 = (__int64)v44;
                  v7 = 16;
                  v15 = *v14;
                  v16 = sub_145CDC0(0x10u, v44);
                  v18 = v16;
                  if ( v16 )
                  {
                    *(_DWORD *)v16 = v13;
                    *(_QWORD *)(v16 + 8) = *(_QWORD *)(v15 + 8);
                  }
                  v19 = *((unsigned int *)v12 + 18);
                  if ( (unsigned int)v19 >= *((_DWORD *)v12 + 19) )
                  {
                    a2 = (__int64)v37;
                    v7 = v35;
                    sub_16CD150(v35, v37, 0, 8, v10, v17);
                    v19 = *((unsigned int *)v12 + 18);
                  }
                  a3 = v12[8];
                  ++v14;
                  *(_QWORD *)(a3 + 8 * v19) = v18;
                  v13 = *((_DWORD *)v12 + 18) + 1;
                  *((_DWORD *)v12 + 18) = v13;
                }
                while ( v43 != v14 );
                v5 = v34;
                v8 = (__int64)v12;
              }
              v20 = *(_QWORD *)v5;
              if ( *(_QWORD *)v5 != *(_QWORD *)v5 + 24LL * *(unsigned int *)(v5 + 8) )
              {
                v21 = &v45;
                v22 = *(unsigned int *)(v8 + 8);
                v23 = *(_QWORD *)v5 + 24LL * *(unsigned int *)(v5 + 8);
                do
                {
                  a2 = *(_QWORD *)v20;
                  v24 = **(unsigned int **)(v20 + 16);
                  v25 = *(_QWORD *)(v8 + 64);
                  v45.m128i_i64[0] = *(_QWORD *)v20;
                  v26 = *(_QWORD *)(v25 + 8 * v24);
                  v27 = *(_QWORD *)(v20 + 8);
                  v46 = v26;
                  v45.m128i_i64[1] = v27;
                  if ( *(_DWORD *)(v8 + 12) <= (unsigned int)v22 )
                  {
                    a2 = (__int64)v38;
                    v7 = v8;
                    sub_16CD150(v8, v38, 0, 24, v10, (int)v21);
                    v22 = *(unsigned int *)(v8 + 8);
                  }
                  v20 += 24;
                  v28 = (__m128i *)(*(_QWORD *)v8 + 24 * v22);
                  a3 = v46;
                  *v28 = _mm_loadu_si128(&v45);
                  v28[1].m128i_i64[0] = a3;
                  v22 = (unsigned int)(*(_DWORD *)(v8 + 8) + 1);
                  *(_DWORD *)(v8 + 8) = v22;
                }
                while ( v23 != v20 );
              }
            }
            *(_DWORD *)(v8 + 112) = v42;
          }
          v29 = *(_QWORD *)(a1 + 104);
          *(_QWORD *)(a1 + 104) = v8;
          *(_QWORD *)(v8 + 104) = v29;
        }
        if ( !*(_QWORD *)(a4 + 16) )
          goto LABEL_32;
        a2 = v8;
        (*(void (__fastcall **)(__int64, __int64))(a4 + 24))(a4, v8);
        v39 &= ~v42;
      }
      v5 = *(_QWORD *)(v5 + 104);
    }
    while ( v5 );
    v4 = v44;
  }
  else
  {
    v39 = a3;
  }
  result = v39;
  if ( v39 )
  {
    v31 = sub_145CDC0(0x78u, v4);
    a2 = v31;
    if ( v31 )
    {
      a3 = v31 + 80;
      *(_QWORD *)(v31 + 96) = 0;
      *(_QWORD *)v31 = v31 + 16;
      *(_QWORD *)(v31 + 8) = 0x200000000LL;
      *(_QWORD *)(v31 + 72) = 0x200000000LL;
      *(_QWORD *)(v31 + 64) = v31 + 80;
      *(_DWORD *)(v31 + 112) = v39;
    }
    v7 = a4;
    v32 = *(_QWORD *)(a1 + 104);
    *(_QWORD *)(a1 + 104) = a2;
    v33 = *(_QWORD *)(a4 + 16) == 0;
    *(_QWORD *)(a2 + 104) = v32;
    if ( v33 )
LABEL_32:
      sub_4263D6(v7, a2, a3);
    return (*(__int64 (**)(void))(a4 + 24))();
  }
  return result;
}
