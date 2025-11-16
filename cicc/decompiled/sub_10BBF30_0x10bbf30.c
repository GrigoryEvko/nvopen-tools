// Function: sub_10BBF30
// Address: 0x10bbf30
//
__int64 __fastcall sub_10BBF30(
        unsigned __int8 *a1,
        unsigned __int8 *a2,
        unsigned __int8 *a3,
        unsigned __int8 a4,
        const __m128i *a5,
        int a6)
{
  __int64 v7; // rax
  unsigned int v9; // r14d
  char v10; // r10
  unsigned int v11; // r12d
  __int64 v12; // rax
  __int64 *v13; // r11
  __int64 *v14; // r13
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __m128i v17; // xmm3
  int v18; // edi
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // r15
  int v22; // ebx
  unsigned int v23; // ebx
  __int64 v24; // rdx
  int v25; // r13d
  __int64 v26; // rbx
  __int64 v27; // r13
  __int64 v28; // rdx
  unsigned int v29; // esi
  char v31; // [rsp+7h] [rbp-C9h]
  __int64 v33; // [rsp+10h] [rbp-C0h]
  __int64 v34; // [rsp+18h] [rbp-B8h]
  char v35[32]; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v36; // [rsp+40h] [rbp-90h]
  __m128i v37[2]; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int64 v38; // [rsp+70h] [rbp-60h]
  unsigned __int8 *v39; // [rsp+78h] [rbp-58h]
  __m128i v40; // [rsp+80h] [rbp-50h]
  __int64 v41; // [rsp+90h] [rbp-40h]

  if ( a2 == a3 )
    return 0;
  if ( a2 == a1 )
    return (__int64)a3;
  if ( (unsigned __int8)(*a1 - 57) > 2u || a6 == 3 )
    return 0;
  v7 = *((_QWORD *)a1 + 2);
  if ( !v7 || (v10 = a4, v9 = a4, *(_QWORD *)(v7 + 8)) )
  {
    v9 = 1;
    v10 = 1;
  }
  v11 = a6 + 1;
  v31 = v10;
  v34 = sub_10BBF30(*((_QWORD *)a1 - 8), a2, a3, v9, a5, (unsigned int)(a6 + 1));
  v12 = sub_10BBF30(*((_QWORD *)a1 - 4), a2, a3, v9, a5, v11);
  v13 = (__int64 *)v34;
  v14 = (__int64 *)v12;
  if ( !(v12 | v34) )
    return 0;
  if ( v34 )
  {
    if ( !v12 )
      v14 = (__int64 *)*((_QWORD *)a1 - 4);
  }
  else
  {
    v13 = (__int64 *)*((_QWORD *)a1 - 8);
  }
  v15 = _mm_loadu_si128(a5 + 6);
  v16 = _mm_loadu_si128(a5 + 7);
  v17 = _mm_loadu_si128(a5 + 9);
  v18 = *a1 - 29;
  v19 = a5[10].m128i_i64[0];
  v38 = _mm_loadu_si128(a5 + 8).m128i_u64[0];
  v41 = v19;
  v39 = a1;
  v33 = (__int64)v13;
  v37[0] = v15;
  v37[1] = v16;
  v40 = v17;
  v20 = (__int64)sub_101E7C0(v18, v13, v14, v37);
  if ( !v20 )
  {
    if ( !v31 )
    {
      v21 = a5[2].m128i_i64[0];
      v22 = *a1;
      v36 = 257;
      v23 = v22 - 29;
      v20 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64 *))(**(_QWORD **)(v21 + 80) + 16LL))(
              *(_QWORD *)(v21 + 80),
              v23,
              v33,
              v14);
      if ( !v20 )
      {
        LOWORD(v38) = 257;
        v20 = sub_B504D0(v23, v33, (__int64)v14, (__int64)v37, 0, 0);
        if ( (unsigned __int8)sub_920620(v20) )
        {
          v24 = *(_QWORD *)(v21 + 96);
          v25 = *(_DWORD *)(v21 + 104);
          if ( v24 )
            sub_B99FD0(v20, 3u, v24);
          sub_B45150(v20, v25);
        }
        (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(v21 + 88) + 16LL))(
          *(_QWORD *)(v21 + 88),
          v20,
          v35,
          *(_QWORD *)(v21 + 56),
          *(_QWORD *)(v21 + 64));
        v26 = *(_QWORD *)v21;
        v27 = *(_QWORD *)v21 + 16LL * *(unsigned int *)(v21 + 8);
        if ( *(_QWORD *)v21 != v27 )
        {
          do
          {
            v28 = *(_QWORD *)(v26 + 8);
            v29 = *(_DWORD *)v26;
            v26 += 16;
            sub_B99FD0(v20, v29, v28);
          }
          while ( v27 != v26 );
        }
      }
      return v20;
    }
    return 0;
  }
  return v20;
}
