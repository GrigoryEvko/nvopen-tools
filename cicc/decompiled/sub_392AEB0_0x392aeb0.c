// Function: sub_392AEB0
// Address: 0x392aeb0
//
__int64 __fastcall sub_392AEB0(__int64 a1, __int64 a2)
{
  _BYTE *v4; // rsi
  __int64 v5; // rax
  _BYTE *v7; // rsi
  _BYTE *v8; // rdx
  _BYTE *v9; // rax
  _BYTE *v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rdx
  _BYTE *v13; // rax
  __m128i *v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 v17[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v18[6]; // [rsp+20h] [rbp-30h] BYREF

  v4 = *(_BYTE **)(a2 + 144);
  if ( *v4 == 42 )
  {
    v7 = v4 + 1;
    v8 = (_BYTE *)(*(_QWORD *)(a2 + 152) + *(_QWORD *)(a2 + 160));
    *(_BYTE *)(a2 + 169) = 0;
    *(_QWORD *)(a2 + 144) = v7;
    v9 = v7;
    if ( v7 != v8 )
    {
      while ( 1 )
      {
        v10 = v9++;
        *(_QWORD *)(a2 + 144) = v9;
        if ( *(v9 - 1) == 42 && *v9 == 47 )
          break;
        if ( v9 == v8 )
          goto LABEL_14;
      }
      v11 = *(_QWORD *)(a2 + 120);
      if ( v11 )
      {
        (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, signed __int64))(*(_QWORD *)v11 + 16LL))(
          v11,
          v7,
          v7,
          v10 - v7);
        v9 = *(_BYTE **)(a2 + 144);
      }
      v12 = *(_QWORD *)(a2 + 104);
      v13 = v9 + 1;
      *(_QWORD *)(a2 + 144) = v13;
      *(_DWORD *)a1 = 7;
      *(_QWORD *)(a1 + 8) = v12;
      *(_QWORD *)(a1 + 16) = &v13[-v12];
      *(_DWORD *)(a1 + 32) = 64;
      *(_QWORD *)(a1 + 24) = 0;
      return a1;
    }
LABEL_14:
    v16 = 20;
    v17[0] = (unsigned __int64)v18;
    v14 = (__m128i *)sub_22409D0((__int64)v17, &v16, 0);
    v17[0] = (unsigned __int64)v14;
    v18[0] = v16;
    *v14 = _mm_load_si128((const __m128i *)&xmmword_3F90110);
    v15 = v17[0];
    v14[1].m128i_i32[0] = 1953391981;
    v17[1] = v16;
    *(_BYTE *)(v15 + v16) = 0;
    sub_392A760(a1, (_QWORD *)a2, *(_QWORD *)(a2 + 104), v17);
    if ( (_QWORD *)v17[0] == v18 )
      return a1;
    j_j___libc_free_0(v17[0]);
    return a1;
  }
  else
  {
    if ( *v4 != 47 )
    {
      *(_BYTE *)(a2 + 169) = 0;
      v5 = *(_QWORD *)(a2 + 104);
      *(_DWORD *)a1 = 15;
      *(_QWORD *)(a1 + 8) = v5;
      *(_QWORD *)(a1 + 16) = 1;
      *(_DWORD *)(a1 + 32) = 64;
      *(_QWORD *)(a1 + 24) = 0;
      return a1;
    }
    *(_QWORD *)(a2 + 144) = v4 + 1;
    sub_392AD60(a1, a2);
    return a1;
  }
}
