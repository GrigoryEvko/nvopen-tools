// Function: sub_33C5C00
// Address: 0x33c5c00
//
__int64 __fastcall sub_33C5C00(__int64 a1, __int64 a2, __int64 *a3, unsigned int *a4)
{
  __int64 v4; // rax
  __int64 v5; // r8
  char v10; // al
  char v11; // r15
  __int64 v12; // rsi
  unsigned __int64 v13; // rcx
  unsigned int v14; // edx
  unsigned int v15; // r9d
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // r10
  __int64 v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __m128i *v23; // r9
  _BYTE **v24; // rax
  unsigned __int64 *v25; // rdx
  __int8 *v26; // rsi
  __int64 i; // rbx
  unsigned int v28; // edi
  __int64 *v29; // [rsp-90h] [rbp-90h]
  __int64 v30; // [rsp-88h] [rbp-88h]
  __int64 v31; // [rsp-80h] [rbp-80h]
  unsigned int v32; // [rsp-80h] [rbp-80h]
  __m128i *v33; // [rsp-80h] [rbp-80h]
  unsigned int v34; // [rsp-6Ch] [rbp-6Ch] BYREF
  __int64 v35; // [rsp-68h] [rbp-68h]
  __m128i *v36; // [rsp-60h] [rbp-60h]
  __m128i *v37; // [rsp-58h] [rbp-58h]
  __int128 v38; // [rsp-50h] [rbp-50h]
  unsigned int v39; // [rsp-40h] [rbp-40h]
  int v40; // [rsp-3Ch] [rbp-3Ch]

  v4 = *(_QWORD *)(a1 + 960);
  v5 = *(_QWORD *)(v4 + 744);
  if ( (unsigned int)qword_5039208 > 0x64 || !*(_QWORD *)(v4 + 32) )
    return *(_QWORD *)(v4 + 744);
  if ( (unsigned __int64)(a3[1] - *a3) > 0x28 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 856) + 648LL) )
    {
      v31 = *(_QWORD *)(v4 + 744);
      v10 = sub_B2D610(**(_QWORD **)(v5 + 32), 18);
      v5 = v31;
      v11 = v10;
      if ( !v10 )
      {
        sub_F02DB0(&v34, qword_5039208, 0x64u);
        v12 = *a3;
        v5 = v31;
        v13 = 0xCCCCCCCCCCCCCCCDLL * ((a3[1] - *a3) >> 3);
        if ( v13 )
        {
          v14 = 0;
          v15 = 0;
          v16 = 0;
          do
          {
            if ( v34 <= *(_DWORD *)(v12 + 40 * v16 + 32) )
            {
              v34 = *(_DWORD *)(v12 + 40 * v16 + 32);
              v15 = v14;
              v11 = 1;
            }
            v16 = ++v14;
          }
          while ( v14 < v13 );
          v32 = v15;
          if ( v11 )
          {
            v17 = *(_QWORD *)(a1 + 960);
            v18 = *(__int64 **)(v5 + 8);
            LOBYTE(v36) = 0;
            v30 = v5;
            v29 = v18;
            v19 = sub_2E7AAE0(*(_QWORD *)(v17 + 8), *(_QWORD *)(v5 + 16), v35, 0);
            sub_2E33BD0(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 8LL) + 320LL, v19);
            v20 = *v29;
            v21 = *(_QWORD *)v19 & 7LL;
            *(_QWORD *)(v19 + 8) = v29;
            v20 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)v19 = v20 | v21;
            *(_QWORD *)(v20 + 8) = v19;
            *v29 = v19 | *v29 & 7;
            sub_33C4170(a1, **(_BYTE ***)(a2 - 8));
            v22 = *a3;
            v38 = 0;
            v40 = 0;
            v23 = (__m128i *)(v22 + 40LL * v32);
            v35 = v30;
            v39 = 0x80000000 - v34;
            v24 = *(_BYTE ***)(a2 - 8);
            v36 = v23;
            v37 = v23;
            v33 = v23;
            sub_33C4220(
              a1,
              *v24,
              v30,
              v19,
              v30,
              (__int64)v23,
              (__m128i)0LL,
              v30,
              v23,
              (unsigned __int64)v23,
              0,
              0,
              0x80000000 - v34);
            v25 = (unsigned __int64 *)a3[1];
            v26 = &v33[2].m128i_i8[8];
            if ( v25 != &v33[2].m128i_u64[1] )
            {
              memmove(v33, v26, (char *)v25 - v26);
              v26 = (__int8 *)a3[1];
            }
            a3[1] = (__int64)(v26 - 40);
            for ( i = *a3; v26 - 40 != (__int8 *)i; *(_DWORD *)(i - 8) = sub_33652A0(v28, v34) )
            {
              v28 = *(_DWORD *)(i + 32);
              i += 40;
            }
            v5 = v19;
            *a4 = v34;
          }
        }
      }
    }
  }
  return v5;
}
