// Function: sub_16BE700
// Address: 0x16be700
//
__int64 __fastcall sub_16BE700(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        void *a5,
        size_t a6,
        unsigned int a7,
        __int64 a8)
{
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __m128i *v17; // rdx
  __int64 v18; // r12
  __m128i si128; // xmm0
  _BYTE *v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rdi
  _BYTE *v26; // rax
  __int64 v27; // rax
  void *src; // [rsp+0h] [rbp-60h] BYREF
  size_t n; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+14h] [rbp-4Ch]
  int v31; // [rsp+1Ch] [rbp-44h]
  _QWORD v32[2]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v33; // [rsp+30h] [rbp-30h]

  v8 = (a4 - a3) >> 4;
  src = a5;
  n = a6;
  LOBYTE(v33) = 0;
  if ( (_BYTE)a7 )
  {
    if ( (unsigned int)sub_16C8EF0(a1, a2, a3, v8, (unsigned int)v32, 0, 0, 0, 0, a8, 0) )
    {
      v22 = sub_16E8CB0(a1, a2, v9);
      v23 = *(_QWORD *)(v22 + 24);
      v24 = v22;
      if ( (unsigned __int64)(*(_QWORD *)(v22 + 16) - v23) <= 6 )
      {
        v24 = sub_16E7EE0(v22, "Error: ", 7);
      }
      else
      {
        *(_DWORD *)v23 = 1869771333;
        *(_WORD *)(v23 + 4) = 14962;
        *(_BYTE *)(v23 + 6) = 32;
        *(_QWORD *)(v22 + 24) += 7LL;
      }
      v25 = sub_16E7EE0(v24, *(const char **)a8, *(_QWORD *)(a8 + 8));
      v26 = *(_BYTE **)(v25 + 24);
      if ( *(_BYTE **)(v25 + 16) == v26 )
      {
        sub_16E7EE0(v25, "\n", 1);
      }
      else
      {
        *v26 = 10;
        ++*(_QWORD *)(v25 + 24);
      }
      return a7;
    }
    else
    {
      v33 = 261;
      v32[0] = &src;
      sub_16C50A0(v32, 1);
      v11 = sub_16E8CB0(v32, 1, v10);
      v12 = *(_QWORD **)(v11 + 24);
      if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 7u )
      {
        sub_16E7EE0(v11, " done. \n", 8);
      }
      else
      {
        *v12 = 0xA202E656E6F6420LL;
        *(_QWORD *)(v11 + 24) += 8LL;
      }
      return 0;
    }
  }
  else
  {
    v14 = sub_16C8E00(a1, a2, a3, v8, (unsigned int)v32, 0, 0, 0, a8, 0);
    v31 = v15;
    v30 = v14;
    v16 = sub_16E8CB0(a1, a2, v15);
    v17 = *(__m128i **)(v16 + 24);
    v18 = v16;
    if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 0x1Du )
    {
      v27 = sub_16E7EE0(v16, "Remember to erase graph file: ", 30);
      v20 = *(_BYTE **)(v27 + 24);
      v18 = v27;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F66B20);
      qmemcpy(&v17[1], "e graph file: ", 14);
      *v17 = si128;
      v20 = (_BYTE *)(*(_QWORD *)(v16 + 24) + 30LL);
      *(_QWORD *)(v16 + 24) = v20;
    }
    v21 = *(_BYTE **)(v18 + 16);
    if ( v21 - v20 < n )
    {
      v18 = sub_16E7EE0(v18, (const char *)src, n);
      v21 = *(_BYTE **)(v18 + 16);
      v20 = *(_BYTE **)(v18 + 24);
    }
    else if ( n )
    {
      memcpy(v20, src, n);
      v21 = *(_BYTE **)(v18 + 16);
      v20 = (_BYTE *)(n + *(_QWORD *)(v18 + 24));
      *(_QWORD *)(v18 + 24) = v20;
    }
    if ( v21 == v20 )
    {
      sub_16E7EE0(v18, "\n", 1);
    }
    else
    {
      *v20 = 10;
      ++*(_QWORD *)(v18 + 24);
    }
    return a7;
  }
}
