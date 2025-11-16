// Function: sub_2092C30
// Address: 0x2092c30
//
__int64 __fastcall sub_2092C30(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        unsigned int *a4,
        __m128i a5,
        __m128i a6,
        __m128i a7)
{
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v14; // rdi
  unsigned __int64 v15; // rsi
  char v16; // r8
  unsigned int v17; // edx
  unsigned int v18; // ecx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  int v22; // r8d
  __int64 v23; // rax
  int v24; // r9d
  __int64 **v25; // r13
  char *v26; // rdx
  char *v27; // rsi
  __int64 i; // rbx
  unsigned int v29; // edi
  __int64 *v30; // [rsp+8h] [rbp-88h]
  unsigned int dest; // [rsp+10h] [rbp-80h]
  char *desta; // [rsp+10h] [rbp-80h]
  __int64 v33; // [rsp+18h] [rbp-78h]
  unsigned int v34; // [rsp+2Ch] [rbp-64h] BYREF
  __int64 v35; // [rsp+30h] [rbp-60h]
  void *src; // [rsp+38h] [rbp-58h]
  __int64 v37; // [rsp+40h] [rbp-50h]
  int v38[4]; // [rsp+48h] [rbp-48h]
  unsigned int v39[14]; // [rsp+58h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 712);
  v8 = *(_QWORD *)(v7 + 784);
  if ( (unsigned int)dword_4FCEE80 <= 0x64 )
  {
    if ( *(_QWORD *)(v7 + 32) )
    {
      if ( (unsigned __int64)(a3[1] - *a3) > 0x28 )
      {
        if ( (unsigned int)sub_1700720(*(_QWORD *)(a1 + 544)) )
        {
          if ( !(unsigned __int8)sub_1560180(**(_QWORD **)(v8 + 56) + 112LL, 17) )
          {
            sub_16AF710(&v34, dword_4FCEE80, 0x64u);
            v14 = *a3;
            v15 = 0xCCCCCCCCCCCCCCCDLL * ((a3[1] - *a3) >> 3);
            if ( v15 )
            {
              v16 = 0;
              v17 = 0;
              v18 = 0;
              v19 = 0;
              do
              {
                if ( v34 <= *(_DWORD *)(v14 + 40 * v19 + 32) )
                {
                  v34 = *(_DWORD *)(v14 + 40 * v19 + 32);
                  v18 = v17;
                  v16 = 1;
                }
                v19 = ++v17;
              }
              while ( v17 < v15 );
              dest = v18;
              if ( v16 )
              {
                v30 = *(__int64 **)(v8 + 8);
                v33 = (__int64)sub_1E0B6F0(*(_QWORD *)(*(_QWORD *)(a1 + 712) + 8LL), *(_QWORD *)(v8 + 40));
                sub_1DD8DC0(*(_QWORD *)(*(_QWORD *)(a1 + 712) + 8LL) + 320LL, v33);
                v20 = *v30;
                *(_QWORD *)(v33 + 8) = v30;
                *(_QWORD *)v33 = v20 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v33 & 7LL;
                *(_QWORD *)((v20 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v33;
                *v30 = v33 | *v30 & 7;
                if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                  v21 = *(__int64 **)(a2 - 8);
                else
                  v21 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
                sub_2090460(a1, *v21, a5, a6, a7);
                v23 = *a3;
                v39[1] = 0;
                v35 = v8;
                v24 = v23 + 40 * dest;
                *(_OWORD *)v38 = 0;
                src = (void *)(v23 + 40LL * dest);
                v37 = (__int64)src;
                v39[0] = 0x80000000 - v34;
                if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                  v25 = *(__int64 ***)(a2 - 8);
                else
                  v25 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
                desta = (char *)(v23 + 40LL * dest);
                sub_20912B0(
                  a1,
                  *v25,
                  v8,
                  v33,
                  v22,
                  v24,
                  (__m128i)0LL,
                  a6,
                  a7,
                  v35,
                  (__m128i *)src,
                  v37,
                  v38[0],
                  v38[2],
                  v39[0]);
                v26 = (char *)a3[1];
                v27 = desta + 40;
                if ( v26 != desta + 40 )
                {
                  memmove(desta, v27, v26 - v27);
                  v27 = (char *)a3[1];
                }
                a3[1] = (__int64)(v27 - 40);
                for ( i = *a3; v27 - 40 != (char *)i; *(_DWORD *)(i - 8) = sub_2044050(v29, v34) )
                {
                  v29 = *(_DWORD *)(i + 32);
                  i += 40;
                }
                v8 = v33;
                *a4 = v34;
              }
            }
          }
        }
      }
    }
  }
  return v8;
}
