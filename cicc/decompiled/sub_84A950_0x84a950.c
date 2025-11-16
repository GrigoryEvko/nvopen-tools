// Function: sub_84A950
// Address: 0x84a950
//
__int64 __fastcall sub_84A950(__int64 a1, __m128i *a2, __int64 a3, int a4, int a5, __m128i *a6)
{
  __int64 v8; // rdi
  __int64 result; // rax
  unsigned int v10; // r9d
  __int64 v11; // [rsp-10h] [rbp-20h]

  if ( !*(_BYTE *)(a1 + 8) )
  {
    v8 = *(_QWORD *)(a1 + 24);
    if ( *(_BYTE *)(v8 + 24) != 5 )
    {
      sub_838020(v8 + 8, 0, a2, a3, a4, a5, a6);
      return v11;
    }
    a1 = *(_QWORD *)(v8 + 152);
  }
  if ( a3 )
  {
    v10 = ((*(_BYTE *)(a3 + 34) & 0x20) != 0) << 6;
    if ( a4 )
      sub_839D30(a1, a2, 0, 0, 0, v10, 0, 0, 0, 0, 0, a6);
    else
      sub_839D30(a1, a2, 0, 0, 0, v10 | 0x2000, 0, 0, 0, 0, 0, a6);
    result = *(unsigned int *)(a3 + 36);
    a6[1].m128i_i32[2] = result;
  }
  else if ( a4 )
  {
    return sub_839D30(a1, a2, 0, 0, 0, 0, 0, 0, 0, 0, 0, a6);
  }
  else
  {
    return sub_839D30(a1, a2, 0, 0, 0, 0x2000u, 0, 0, 0, 0, 0, a6);
  }
  return result;
}
