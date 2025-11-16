// Function: sub_1BC9810
// Address: 0x1bc9810
//
void __fastcall sub_1BC9810(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // r14
  __int64 v4; // rdx
  __int64 i; // r15
  __int64 v6; // rdx
  __int64 *v7; // r13
  __int64 v8; // rax
  __int64 *v9; // rdx
  _QWORD *v10; // r15
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rdi
  unsigned __int64 *v14; // rdi
  unsigned __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // r15
  __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 *j; // r15
  __int64 v21; // rsi
  __int64 *v22; // r15
  __int64 *k; // rcx
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 *v27; // [rsp-A0h] [rbp-A0h]
  __int64 *v28; // [rsp-A0h] [rbp-A0h]
  __int64 *v29; // [rsp-A0h] [rbp-A0h]
  int v30; // [rsp-90h] [rbp-90h] BYREF
  int v31; // [rsp-8Ch] [rbp-8Ch] BYREF
  _QWORD v32[4]; // [rsp-88h] [rbp-88h] BYREF
  __int64 v33; // [rsp-68h] [rbp-68h] BYREF
  int v34; // [rsp-60h] [rbp-60h] BYREF
  __int64 v35; // [rsp-58h] [rbp-58h]
  int *v36; // [rsp-50h] [rbp-50h]
  int *v37; // [rsp-48h] [rbp-48h]
  __int64 v38; // [rsp-40h] [rbp-40h]

  if ( a2[23] )
  {
    sub_1BC9790((__int64)a2);
    v34 = 0;
    v3 = (_QWORD *)a2[23];
    v35 = 0;
    v36 = &v34;
    v37 = &v34;
    v38 = 0;
    v30 = 0;
    v31 = 0;
    if ( v3 != (_QWORD *)a2[24] )
    {
      do
      {
        v32[0] = a1;
        v32[2] = &v31;
        v32[1] = &v30;
        v32[3] = a2;
        sub_1BC9610((__int64)a2, (__int64)v3, (void (__fastcall *)(__int64, __int64))sub_1BCAB60, (__int64)v32);
        v4 = v3[4];
        if ( v4 == v3[5] + 40LL || !v4 )
          v3 = 0;
        else
          v3 = (_QWORD *)(v4 - 24);
      }
      while ( (_QWORD *)a2[24] != v3 );
      for ( i = a2[23]; (_QWORD *)i != v3; v3 = (_QWORD *)a2[24] )
      {
        v32[0] = &v33;
        sub_1BC9610((__int64)a2, i, (void (__fastcall *)(__int64, __int64))sub_1BBA0A0, (__int64)v32);
        v6 = *(_QWORD *)(i + 32);
        if ( v6 == *(_QWORD *)(i + 40) + 40LL || !v6 )
          i = 0;
        else
          i = v6 - 24;
      }
      for ( ; v38; --v31 )
      {
        v7 = (__int64 *)*((_QWORD *)v36 + 4);
        v8 = sub_220F330(v36, &v34);
        j_j___libc_free_0(v8, 40);
        --v38;
        if ( v7 )
        {
          v9 = v7;
          do
          {
            v10 = v3;
            v3 = (_QWORD *)*v9;
            v11 = v10[4];
            if ( v11 == v10[5] + 40LL || !v11 )
              v12 = 0;
            else
              v12 = (_QWORD *)(v11 - 24);
            if ( v3 != v12 )
            {
              v27 = v9;
              v13 = *a2 + 40LL;
              if ( !v3 )
              {
                sub_157EA20(v13, -24);
                BUG();
              }
              sub_157EA20(v13, (__int64)v3);
              v14 = (unsigned __int64 *)v3[4];
              v15 = v3[3] & 0xFFFFFFFFFFFFFFF8LL;
              *v14 = v15 | *v14 & 7;
              *(_QWORD *)(v15 + 8) = v14;
              v3[3] &= 7uLL;
              v3[4] = 0;
              sub_157E9D0(*a2 + 40LL, (__int64)v3);
              v16 = v10[3];
              v3[4] = v10 + 3;
              v16 &= 0xFFFFFFFFFFFFFFF8LL;
              v9 = v27;
              v3[3] = v16 | v3[3] & 7LL;
              *(_QWORD *)(v16 + 8) = v3 + 3;
              v10[3] = v10[3] & 7LL | (unsigned __int64)(v3 + 3);
            }
            v9 = (__int64 *)v9[2];
          }
          while ( v9 );
        }
        *((_BYTE *)v7 + 100) = 1;
        do
        {
          v17 = *v7;
          if ( *v7 == v7[13] )
          {
            v18 = 3LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
            if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
            {
              v19 = *(__int64 **)(v17 - 8);
              v28 = &v19[v18];
            }
            else
            {
              v28 = (__int64 *)*v7;
              v19 = (__int64 *)(v17 - v18 * 8);
            }
            for ( j = v19; v28 != j; j += 3 )
            {
              v21 = *j;
              if ( *(_BYTE *)(*j + 16) > 0x17u )
              {
                v32[0] = &v33;
                sub_1BC9610((__int64)a2, v21, (void (__fastcall *)(__int64, __int64))sub_1BBA060, (__int64)v32);
              }
            }
            v22 = (__int64 *)v7[4];
            for ( k = &v22[*((unsigned int *)v7 + 10)]; k != v22; ++v22 )
            {
              v24 = *v22;
              v25 = *(_QWORD *)(*v22 + 8);
              --*(_DWORD *)(*v22 + 92);
              if ( (*(_DWORD *)(v25 + 96))-- == 1 )
              {
                v29 = k;
                v32[0] = *(_QWORD *)(v24 + 8);
                sub_1BB95B0((__int64)&v33, (__int64)v32);
                k = v29;
              }
            }
          }
          v7 = (__int64 *)v7[2];
        }
        while ( v7 );
      }
    }
    a2[23] = 0;
    sub_1BBB070(v35);
  }
}
