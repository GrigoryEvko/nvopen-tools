// Function: sub_17040B0
// Address: 0x17040b0
//
__int64 __fastcall sub_17040B0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14)
{
  int v14; // r8d
  __int64 v15; // r15
  __int64 v17; // r14
  __int64 *v18; // rsi
  __int64 i; // rdi
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r11
  __int64 v26; // rbx
  __int64 j; // r13
  __int64 v28; // rax
  int v29; // edx
  unsigned int v30; // r13d
  __int64 v31; // rdx
  __int64 *v32; // rax
  double v33; // xmm4_8
  double v34; // xmm5_8
  unsigned int v35; // eax
  int v37; // ebx

  v14 = 0;
  v15 = a2 + 72;
  v17 = *(_QWORD *)(a2 + 80);
  v18 = (__int64 *)(a1 + 40);
  for ( i = a1 + 24; v15 != v17; v17 = *(_QWORD *)(v17 + 8) )
  {
    v20 = *(_QWORD *)(a1 + 16);
    v21 = v17 - 24;
    if ( !v17 )
      v21 = 0;
    v22 = *(unsigned int *)(v20 + 48);
    if ( (_DWORD)v22 )
    {
      a14 = *(_QWORD *)(v20 + 32);
      v23 = (v22 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v24 = (__int64 *)(a14 + 16LL * v23);
      v25 = *v24;
      if ( v21 == *v24 )
      {
LABEL_6:
        if ( v24 != (__int64 *)(a14 + 16 * v22) )
        {
          if ( v24[1] )
          {
            v26 = *(_QWORD *)(v21 + 48);
            for ( j = v21 + 40; j != v26; v26 = *(_QWORD *)(v26 + 8) )
            {
              if ( !v26 )
                BUG();
              if ( *(_BYTE *)(v26 - 8) == 60 )
              {
                v28 = *(unsigned int *)(a1 + 32);
                if ( (unsigned int)v28 >= *(_DWORD *)(a1 + 36) )
                {
                  sub_16CD150(i, v18, 0, 8, 0, a14);
                  v14 = 0;
                  v28 = *(unsigned int *)(a1 + 32);
                }
                *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v28) = v26 - 24;
                ++*(_DWORD *)(a1 + 32);
              }
            }
          }
        }
      }
      else
      {
        v29 = 1;
        while ( v25 != -8 )
        {
          v37 = v29 + 1;
          v23 = (v22 - 1) & (v29 + v23);
          v24 = (__int64 *)(a14 + 16LL * v23);
          v25 = *v24;
          if ( v21 == *v24 )
            goto LABEL_6;
          v29 = v37;
        }
      }
    }
  }
  v30 = 0;
  while ( 1 )
  {
    v35 = *(_DWORD *)(a1 + 32);
    if ( !v35 )
      break;
    v31 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * v35 - 8);
    *(_DWORD *)(a1 + 32) = v35 - 1;
    *(_QWORD *)(a1 + 72) = v31;
    v32 = (__int64 *)sub_1703ED0(a1, (__int64)v18, v31, v35, v14, a14);
    if ( v32 )
    {
      v18 = v32;
      v30 = 1;
      sub_1702220(a1, v32, a3, a4, a5, a6, v33, v34, a9, a10);
    }
  }
  return v30;
}
