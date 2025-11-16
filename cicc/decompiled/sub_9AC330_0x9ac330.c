// Function: sub_9AC330
// Address: 0x9ac330
//
__int64 __fastcall sub_9AC330(__int64 a1, __int64 a2, unsigned int a3, __m128i *a4)
{
  unsigned int v6; // ebx
  __int64 v8; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 8);
  v6 = sub_BCB060(v8);
  if ( !v6 )
    v6 = sub_AE43A0(a4->m128i_i64[0], v8);
  *(_DWORD *)(a1 + 8) = v6;
  if ( v6 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    *(_DWORD *)(a1 + 24) = v6;
    sub_C43690(a1 + 16, 0, 0);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = 0;
  }
  sub_9AC0E0(a2, (unsigned __int64 *)a1, a3, a4);
  return a1;
}
