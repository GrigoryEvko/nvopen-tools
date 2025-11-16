// Function: sub_9B0110
// Address: 0x9b0110
//
__int64 __fastcall sub_9B0110(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __m128i *a5)
{
  unsigned int v7; // ebx
  __int64 v9; // [rsp+0h] [rbp-40h]

  v9 = *(_QWORD *)(a2 + 8);
  v7 = sub_BCB060(v9);
  if ( !v7 )
    v7 = sub_AE43A0(a5->m128i_i64[0], v9);
  *(_DWORD *)(a1 + 8) = v7;
  if ( v7 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    *(_DWORD *)(a1 + 24) = v7;
    sub_C43690(a1 + 16, 0, 0);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = 0;
  }
  sub_9AB8E0((unsigned __int8 *)a2, a3, (unsigned __int64 *)a1, a4, a5);
  return a1;
}
