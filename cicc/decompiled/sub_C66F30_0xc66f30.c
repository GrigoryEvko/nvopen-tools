// Function: sub_C66F30
// Address: 0xc66f30
//
__int64 __fastcall sub_C66F30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        const void *a5,
        size_t a6,
        unsigned int a7,
        _QWORD *a8)
{
  __int64 v8; // rcx
  __int64 v11; // rax
  _QWORD *v12; // rdx
  __int64 v14; // rax
  __m128i *v15; // rdx
  __int64 v16; // r13
  __m128i si128; // xmm0
  _BYTE *v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rdi
  _BYTE *v24; // rax
  __int64 v25; // rax
  __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  size_t v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+20h] [rbp-40h]
  __int16 v29; // [rsp+30h] [rbp-30h]

  v8 = (a4 - a3) >> 4;
  LOBYTE(v28) = 0;
  if ( (_BYTE)a7 )
  {
    if ( (unsigned int)sub_C881F0(a1, a2, a3, v8, 0, 0, v26, v27, v28, 0, 0, (__int64)a8, 0, 0) )
    {
      v20 = sub_CB72A0(a1, a2);
      v21 = *(_QWORD *)(v20 + 32);
      v22 = v20;
      if ( (unsigned __int64)(*(_QWORD *)(v20 + 24) - v21) <= 6 )
      {
        v22 = sub_CB6200(v20, "Error: ", 7);
      }
      else
      {
        *(_DWORD *)v21 = 1869771333;
        *(_WORD *)(v21 + 4) = 14962;
        *(_BYTE *)(v21 + 6) = 32;
        *(_QWORD *)(v20 + 32) += 7LL;
      }
      v23 = sub_CB6200(v22, *a8, a8[1]);
      v24 = *(_BYTE **)(v23 + 32);
      if ( *(_BYTE **)(v23 + 24) == v24 )
      {
        sub_CB6200(v23, "\n", 1);
      }
      else
      {
        *v24 = 10;
        ++*(_QWORD *)(v23 + 32);
      }
      return a7;
    }
    else
    {
      v26 = (__int64)a5;
      v29 = 261;
      v27 = a6;
      sub_C823F0(&v26, 1);
      v11 = sub_CB72A0(&v26, 1);
      v12 = *(_QWORD **)(v11 + 32);
      if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 7u )
      {
        sub_CB6200(v11, " done. \n", 8);
      }
      else
      {
        *v12 = 0xA202E656E6F6420LL;
        *(_QWORD *)(v11 + 32) += 8LL;
      }
      return 0;
    }
  }
  else
  {
    sub_C882E0(a1, a2, a3, v8, 0, 0, v26, v27, v28, 0, (__int64)a8, 0, 0, 0);
    v14 = sub_CB72A0(a1, a2);
    v15 = *(__m128i **)(v14 + 32);
    v16 = v14;
    if ( *(_QWORD *)(v14 + 24) - (_QWORD)v15 <= 0x1Du )
    {
      v25 = sub_CB6200(v14, "Remember to erase graph file: ", 30);
      v18 = *(_BYTE **)(v25 + 32);
      v16 = v25;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F66B20);
      qmemcpy(&v15[1], "e graph file: ", 14);
      *v15 = si128;
      v18 = (_BYTE *)(*(_QWORD *)(v14 + 32) + 30LL);
      *(_QWORD *)(v14 + 32) = v18;
    }
    v19 = *(_BYTE **)(v16 + 24);
    if ( v19 - v18 < a6 )
    {
      v16 = sub_CB6200(v16, a5, a6);
      v19 = *(_BYTE **)(v16 + 24);
      v18 = *(_BYTE **)(v16 + 32);
    }
    else if ( a6 )
    {
      memcpy(v18, a5, a6);
      v19 = *(_BYTE **)(v16 + 24);
      v18 = (_BYTE *)(a6 + *(_QWORD *)(v16 + 32));
      *(_QWORD *)(v16 + 32) = v18;
    }
    if ( v18 == v19 )
    {
      sub_CB6200(v16, "\n", 1);
    }
    else
    {
      *v18 = 10;
      ++*(_QWORD *)(v16 + 32);
    }
    return a7;
  }
}
