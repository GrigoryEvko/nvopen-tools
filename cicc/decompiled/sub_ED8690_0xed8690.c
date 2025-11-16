// Function: sub_ED8690
// Address: 0xed8690
//
__int64 __fastcall sub_ED8690(__int64 a1, int *a2, size_t a3)
{
  __int64 v5; // [rsp-D8h] [rbp-D8h] BYREF
  _QWORD v6[25]; // [rsp-C8h] [rbp-C8h] BYREF

  if ( *(_DWORD *)(a1 + 24) )
    BUG();
  sub_C7D030(v6);
  sub_C7D280((int *)v6, a2, a3);
  sub_C7D290(v6, &v5);
  return v5;
}
