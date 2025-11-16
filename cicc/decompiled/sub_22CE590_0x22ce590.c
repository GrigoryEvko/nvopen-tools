// Function: sub_22CE590
// Address: 0x22ce590
//
void __fastcall sub_22CE590(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  __int64 v7; // rbx
  int v8; // eax
  __m128i *v9; // rdx
  _BYTE *v10; // r8
  __m128i si128; // xmm0
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // [rsp+0h] [rbp-70h]
  __int64 v19; // [rsp+8h] [rbp-68h]
  _BYTE v20[8]; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 v21; // [rsp+18h] [rbp-58h]
  unsigned int v22; // [rsp+20h] [rbp-50h]
  unsigned __int64 v23; // [rsp+28h] [rbp-48h]
  unsigned int v24; // [rsp+30h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 72);
  if ( (*(_BYTE *)(v6 + 2) & 1) != 0 )
  {
    sub_B2C6D0(*(_QWORD *)(a2 + 72), a2, a3, a4);
    v7 = *(_QWORD *)(v6 + 96);
    v18 = v7 + 40LL * *(_QWORD *)(v6 + 104);
    if ( (*(_BYTE *)(v6 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v6, a2, v16, v17);
      v7 = *(_QWORD *)(v6 + 96);
    }
  }
  else
  {
    v7 = *(_QWORD *)(v6 + 96);
    v18 = v7 + 40LL * *(_QWORD *)(v6 + 104);
  }
  if ( v18 != v7 )
  {
    while ( 1 )
    {
      sub_22CDEF0((__int64)v20, *(_QWORD *)(a1 + 8), v7, a2, 0);
      if ( !v20[0] )
        goto LABEL_6;
      v9 = *(__m128i **)(a3 + 32);
      v10 = (_BYTE *)a3;
      if ( *(_QWORD *)(a3 + 24) - (_QWORD)v9 <= 0x12u )
      {
        v10 = (_BYTE *)sub_CB6200(a3, "; LatticeVal for: '", 0x13u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_428A450);
        v9[1].m128i_i8[2] = 39;
        v9[1].m128i_i16[0] = 8250;
        *v9 = si128;
        *(_QWORD *)(a3 + 32) += 19LL;
      }
      v19 = (__int64)v10;
      sub_A69870(v7, v10, 0);
      v12 = v19;
      v13 = *(_QWORD *)(v19 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v19 + 24) - v13) <= 5 )
      {
        v12 = sub_CB6200(v19, "' is: ", 6u);
      }
      else
      {
        *(_DWORD *)v13 = 1936269351;
        *(_WORD *)(v13 + 4) = 8250;
        *(_QWORD *)(v19 + 32) += 6LL;
      }
      v14 = sub_22EAFB0(v12, v20);
      v15 = *(_BYTE **)(v14 + 32);
      if ( *(_BYTE **)(v14 + 24) == v15 )
      {
        sub_CB6200(v14, (unsigned __int8 *)"\n", 1u);
        if ( (unsigned int)v20[0] - 4 <= 1 )
          goto LABEL_14;
LABEL_6:
        v7 += 40;
        if ( v18 == v7 )
          return;
      }
      else
      {
        *v15 = 10;
        v8 = v20[0];
        ++*(_QWORD *)(v14 + 32);
        if ( (unsigned int)(v8 - 4) > 1 )
          goto LABEL_6;
LABEL_14:
        if ( v24 > 0x40 && v23 )
          j_j___libc_free_0_0(v23);
        if ( v22 <= 0x40 || !v21 )
          goto LABEL_6;
        j_j___libc_free_0_0(v21);
        v7 += 40;
        if ( v18 == v7 )
          return;
      }
    }
  }
}
