// Function: sub_21D7B80
// Address: 0x21d7b80
//
__int64 *__fastcall sub_21D7B80(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        __m128i a9,
        int a10)
{
  __int64 v10; // rdx
  char v11; // al
  __int64 *result; // rax
  int v13; // edx
  __int64 v19; // rax
  int v20; // ecx
  int v21; // edx
  __int64 v22; // rsi
  int v23; // edx
  unsigned __int64 *v24; // rcx
  int v25; // eax
  unsigned __int64 v26; // rdi
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int128 *v30; // r8
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rax
  __int128 *v35; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 *v38; // [rsp+8h] [rbp-58h]
  unsigned int v39; // [rsp+10h] [rbp-50h]
  const void **v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+20h] [rbp-40h] BYREF
  int v42; // [rsp+28h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v11 = *(_BYTE *)v10;
  LOBYTE(v39) = *(_BYTE *)v10;
  v40 = *(const void ***)(v10 + 8);
  if ( !*(_BYTE *)v10 || (unsigned __int8)(v11 - 14) <= 0x5Fu )
    return 0;
  v13 = *(unsigned __int16 *)(a2 + 24);
  if ( v13 == 54 )
  {
    if ( v11 != 5 )
      return 0;
    if ( !a10 )
      return 0;
    v31 = *(_QWORD *)(a2 + 48);
    if ( !v31 || *(_QWORD *)(v31 + 32) )
      return 0;
    v32 = *(_QWORD *)(a2 + 32);
    v33 = *(_QWORD *)(a1 + 72);
    v41 = v33;
    if ( v33 )
    {
      v37 = v32;
      sub_1623A60((__int64)&v41, v33, 2);
      v32 = v37;
    }
    v42 = *(_DWORD *)(a1 + 64);
    result = sub_1D3A900(
               (__int64 *)a6,
               0x129u,
               (__int64)&v41,
               v39,
               v40,
               0,
               a7,
               a8,
               a9,
               *(_QWORD *)v32,
               *(__int16 **)(v32 + 8),
               *(_OWORD *)(v32 + 40),
               a4,
               a5);
  }
  else
  {
    if ( v13 != 78 || (unsigned __int8)(v11 - 9) > 1u || !sub_21D7B30(*(_QWORD *)(a6 + 16), *(_QWORD **)(a6 + 32), a10) )
      return 0;
    v19 = *(_QWORD *)(a2 + 48);
    if ( !v19 )
      goto LABEL_35;
    v20 = 0;
    v21 = 0;
    do
    {
      v22 = *(_QWORD *)(v19 + 16);
      ++v21;
      v19 = *(_QWORD *)(v19 + 32);
      v20 += *(_WORD *)(v22 + 24) != 76;
    }
    while ( v19 );
    if ( v21 > 4 )
      return 0;
    if ( v20 )
    {
      v23 = *(_DWORD *)(a1 + 64);
      if ( v23 - *(_DWORD *)(a2 + 64) <= 499 )
        return 0;
      v24 = *(unsigned __int64 **)(a2 + 32);
      v25 = *(unsigned __int16 *)(*v24 + 24);
      if ( v25 != 32 && v25 != 10 )
      {
        v26 = v24[5];
        v27 = *(unsigned __int16 *)(v26 + 24);
        if ( v27 != 10 && v27 != 32 )
        {
          v28 = *(_QWORD *)(*v24 + 48);
          if ( v28 )
          {
            while ( v23 >= *(_DWORD *)(*(_QWORD *)(v28 + 16) + 64LL) )
            {
              v28 = *(_QWORD *)(v28 + 32);
              if ( !v28 )
                goto LABEL_36;
            }
          }
          else
          {
LABEL_36:
            v34 = *(_QWORD *)(v26 + 48);
            if ( !v34 )
              return 0;
            while ( v23 >= *(_DWORD *)(*(_QWORD *)(v34 + 16) + 64LL) )
            {
              v34 = *(_QWORD *)(v34 + 32);
              if ( !v34 )
                return 0;
            }
          }
        }
      }
    }
    else
    {
LABEL_35:
      v24 = *(unsigned __int64 **)(a2 + 32);
    }
    v29 = *(_QWORD *)(a1 + 72);
    v30 = (__int128 *)(v24 + 5);
    v41 = v29;
    if ( v29 )
    {
      v35 = (__int128 *)(v24 + 5);
      v36 = v24;
      sub_1623A60((__int64)&v41, v29, 2);
      v30 = v35;
      v24 = v36;
    }
    v42 = *(_DWORD *)(a1 + 64);
    result = sub_1D3A900(
               (__int64 *)a6,
               0x63u,
               (__int64)&v41,
               v39,
               v40,
               0,
               a7,
               a8,
               a9,
               *v24,
               (__int16 *)v24[1],
               *v30,
               a4,
               a5);
  }
  if ( v41 )
  {
    v38 = result;
    sub_161E7C0((__int64)&v41, v41);
    return v38;
  }
  return result;
}
