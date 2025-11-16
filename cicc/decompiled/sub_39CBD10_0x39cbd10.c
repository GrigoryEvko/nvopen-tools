// Function: sub_39CBD10
// Address: 0x39cbd10
//
__m128i *__fastcall sub_39CBD10(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 *v6; // r13
  __m128i *v7; // r14
  __int64 v9; // rdi
  __int64 v10; // rdx
  char *v11; // r13
  size_t v12; // r10
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdi
  size_t v17; // rdx
  __int64 v18; // [rsp+0h] [rbp-40h]

  v6 = sub_39A81B0((__int64)a1, *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8)));
  v7 = (__m128i *)sub_39A23D0((__int64)a1, (unsigned __int8 *)a2);
  if ( !v7 )
  {
    v7 = (__m128i *)sub_39A5A90((__int64)a1, 26, (__int64)v6, (unsigned __int8 *)a2);
    v9 = *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
    if ( v9 && (sub_161E970(v9), v10) )
    {
      v16 = *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
      if ( v16 )
      {
        v16 = sub_161E970(v16);
        v12 = v17;
      }
      else
      {
        v12 = 0;
      }
      v11 = (char *)v16;
    }
    else
    {
      v11 = "_BLNK_";
      v12 = 6;
    }
    v18 = v12;
    sub_39A3F30(a1, (__int64)v7, 3, v11, v12);
    sub_39C8580((__int64)a1, v11, v18, (__int64)v7, *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8)));
    v13 = *(unsigned int *)(a2 + 8);
    v14 = *(_QWORD *)(a2 + 8 * (3 - v13));
    if ( v14 )
    {
      sub_39A36D0((__int64)a1, (__int64)v7, *(_DWORD *)(a2 + 24), v14);
      v13 = *(unsigned int *)(a2 + 8);
    }
    v15 = *(_QWORD *)(a2 + 8 * (1 - v13));
    if ( v15 )
      sub_39CB550(a1, v7, v15, a3, a4);
  }
  return v7;
}
