// Function: sub_1EB8760
// Address: 0x1eb8760
//
__int64 __fastcall sub_1EB8760(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, unsigned int a5)
{
  __int16 v6; // r14
  int v9; // esi
  __int64 v10; // r9
  __int64 v11; // r10
  unsigned int v12; // ecx
  _BYTE *v13; // r10
  unsigned int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int16 v19; // si
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __m128i *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // eax
  __int64 v30; // [rsp-8h] [rbp-68h]
  unsigned int v31; // [rsp+4h] [rbp-5Ch]
  __int64 v32; // [rsp+8h] [rbp-58h]
  unsigned __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  __m128i v35; // [rsp+10h] [rbp-50h] BYREF
  __int64 v36; // [rsp+20h] [rbp-40h]

  v6 = a3;
  v9 = a4 & 0x7FFFFFFF;
  v10 = a4 & 0x7FFFFFFF;
  v11 = *(_QWORD *)(a1 + 600);
  v35.m128i_i64[1] = a4;
  v12 = *(_DWORD *)(a1 + 400);
  v35.m128i_i64[0] = 0;
  v13 = (_BYTE *)(v10 + v11);
  v14 = (unsigned __int8)*v13;
  LOBYTE(v36) = 0;
  if ( v14 >= v12 )
    goto LABEL_22;
  v15 = *(_QWORD *)(a1 + 392);
  while ( 1 )
  {
    v16 = v15 + 24LL * v14;
    if ( v9 == (*(_DWORD *)(v16 + 8) & 0x7FFFFFFF) )
      break;
    v14 += 256;
    if ( v12 <= v14 )
      goto LABEL_22;
  }
  if ( v16 == v15 + 24LL * v12 )
  {
LABEL_22:
    *v13 = v12;
    v25 = *(unsigned int *)(a1 + 400);
    if ( (unsigned int)v25 >= *(_DWORD *)(a1 + 404) )
    {
      v31 = a5;
      v34 = v10;
      sub_16CD150(a1 + 392, (const void *)(a1 + 408), 0, 24, a5, v10);
      v25 = *(unsigned int *)(a1 + 400);
      a5 = v31;
      v10 = v34;
    }
    v32 = v10;
    v26 = (__m128i *)(*(_QWORD *)(a1 + 392) + 24 * v25);
    v27 = v36;
    *v26 = _mm_loadu_si128(&v35);
    v26[1].m128i_i64[0] = v27;
    v28 = (unsigned int)(*(_DWORD *)(a1 + 400) + 1);
    *(_DWORD *)(a1 + 400) = v28;
    v16 = sub_1EB8380(a1, a2, *(_QWORD *)(a1 + 392) + 24 * v28 - 24, a5);
    v33 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 240) + 24LL) + 16 * v32) & 0xFFFFFFFFFFFFFFF8LL;
    v29 = sub_1EB6550((__int64 *)a1, a4, v33);
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64, _QWORD, _QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 256)
                                                                                              + 416LL))(
      *(_QWORD *)(a1 + 256),
      *(_QWORD *)(a1 + 360),
      a2,
      *(unsigned __int16 *)(v16 + 12),
      v29,
      v33,
      *(_QWORD *)(a1 + 248));
    v17 = v30;
  }
  else
  {
    v17 = *(_QWORD *)(a2 + 32) + 40LL * a3;
    if ( *(_BYTE *)(v16 + 16) )
    {
      v21 = *(unsigned int *)(v17 + 8);
      v22 = *(_DWORD *)(v17 + 8) & 0x7FFFFFFF;
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 368) + 4 * v22) == -1 )
      {
        v23 = *(_QWORD *)(a1 + 240);
        v24 = (int)v21 < 0
            ? *(_QWORD *)(*(_QWORD *)(v23 + 24) + 16 * v22 + 8)
            : *(_QWORD *)(*(_QWORD *)(v23 + 272) + 8 * v21);
        if ( v24 )
        {
          if ( (*(_BYTE *)(v24 + 4) & 8) != 0 )
          {
            while ( 1 )
            {
              v24 = *(_QWORD *)(v24 + 32);
              if ( !v24 )
                break;
              if ( (*(_BYTE *)(v24 + 4) & 8) == 0 )
                goto LABEL_17;
            }
          }
          else
          {
LABEL_17:
            if ( v17 == v24 )
            {
              v18 = *(_QWORD *)(v17 + 32);
              if ( !v18 )
              {
LABEL_21:
                *(_BYTE *)(v24 + 3) |= 0x40u;
                goto LABEL_9;
              }
              while ( (*(_BYTE *)(v18 + 4) & 8) != 0 )
              {
                v18 = *(_QWORD *)(v18 + 32);
                if ( !v18 )
                  goto LABEL_21;
              }
            }
          }
        }
      }
    }
    v18 = (*(_BYTE *)(v17 + 3) & 0x40) != 0;
    if ( ((unsigned __int8)v18 & ((*(_BYTE *)(v17 + 3) >> 4) ^ 1)) != 0
      || ((unsigned __int8)v18 & ((*(_BYTE *)(v17 + 3) & 0x10) != 0)) != 0 )
    {
      *(_BYTE *)(v17 + 3) &= ~0x40u;
    }
  }
LABEL_9:
  *(_QWORD *)v16 = a2;
  v19 = *(_WORD *)(v16 + 12);
  *(_WORD *)(v16 + 14) = v6;
  sub_1EB6840(a1, v19, v17, v18, a5, v10);
  return v16;
}
