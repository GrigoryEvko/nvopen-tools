// Function: sub_13E6440
// Address: 0x13e6440
//
__int64 __fastcall sub_13E6440(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // r14

  v1 = (__int64 *)(a1 + 160);
  if ( !*(_BYTE *)(a1 + 168) )
  {
    v3 = *(_QWORD *)(a1 + 184);
    v4 = *(_QWORD *)(a1 + 192);
    v5 = *(_QWORD *)(v3 + 160);
    if ( !*(_BYTE *)(v5 + 408) )
    {
      sub_137CAE0(*(_QWORD *)(v3 + 160), *(__int64 **)(v5 + 416), *(_QWORD *)(v5 + 424), *(_QWORD **)(v5 + 432));
      *(_BYTE *)(v5 + 408) = 1;
    }
    sub_1370060(v1, *(const void **)(a1 + 176), v5, v4);
    *(_BYTE *)(a1 + 168) = 1;
  }
  return sub_1368E20(v1);
}
