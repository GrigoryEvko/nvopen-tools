// Function: sub_18437D0
// Address: 0x18437d0
//
void __fastcall sub_18437D0(_QWORD *a1, __m128i *a2, int a3, __int64 a4)
{
  const __m128i *v6; // r15
  _QWORD *v7; // rbx
  __int64 v8; // r14
  __m128i v9; // xmm1
  __int64 v10; // rax
  __m128i v11; // xmm3
  _QWORD *v12; // rdx
  __int64 v13; // rsi
  unsigned __int64 v14; // rdi
  unsigned int v15; // eax
  _QWORD *v16; // rax
  unsigned __int64 v17; // rcx
  unsigned int v18; // r8d
  unsigned int v19; // eax
  __m128i v20; // [rsp+0h] [rbp-50h] BYREF
  __m128i v21[4]; // [rsp+10h] [rbp-40h] BYREF

  if ( !a3 )
  {
    sub_1843600(a1, a2);
    return;
  }
  if ( a3 == 1 )
  {
    v6 = *(const __m128i **)a4;
    v7 = a1 + 1;
    v8 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
    if ( *(_QWORD *)a4 != v8 )
    {
      do
      {
        v9 = _mm_loadu_si128(a2);
        v20 = _mm_loadu_si128(v6);
        v21[0] = v9;
        v10 = sub_22077B0(64);
        v11 = _mm_loadu_si128(v21);
        v12 = (_QWORD *)a1[2];
        v13 = v10;
        *(__m128i *)(v10 + 32) = _mm_loadu_si128(&v20);
        *(__m128i *)(v10 + 48) = v11;
        if ( !v12 )
        {
          v12 = v7;
          goto LABEL_25;
        }
        v14 = *(_QWORD *)(v10 + 32);
        while ( 1 )
        {
          v17 = v12[4];
          if ( v17 > v14 )
            break;
          if ( v17 == v14 )
          {
            v15 = *((_DWORD *)v12 + 10);
            if ( *(_DWORD *)(v13 + 40) < v15
              || *(_DWORD *)(v13 + 40) == v15 && *(_BYTE *)(v13 + 44) < *((_BYTE *)v12 + 44) )
            {
              break;
            }
          }
          v16 = (_QWORD *)v12[3];
          if ( !v16 )
            goto LABEL_14;
LABEL_11:
          v12 = v16;
        }
        v16 = (_QWORD *)v12[2];
        if ( v16 )
          goto LABEL_11;
LABEL_14:
        v18 = 1;
        if ( v7 != v12 && v17 <= v14 )
        {
          v18 = 0;
          if ( v17 == v14 )
          {
            v19 = *((_DWORD *)v12 + 10);
            if ( *(_DWORD *)(v13 + 40) >= v19
              && (*(_DWORD *)(v13 + 40) != v19 || *(_BYTE *)(v13 + 44) >= *((_BYTE *)v12 + 44)) )
            {
              v18 = 0;
              goto LABEL_16;
            }
LABEL_25:
            v18 = 1;
          }
        }
LABEL_16:
        ++v6;
        sub_220F040(v18, v13, v12, v7);
        ++a1[5];
      }
      while ( (const __m128i *)v8 != v6 );
    }
  }
}
