// Function: sub_1027660
// Address: 0x1027660
//
__int64 __fastcall sub_1027660(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // r14

  v1 = (__int64 *)(a1 + 176);
  if ( !*(_BYTE *)(a1 + 184) )
  {
    v3 = *(_QWORD *)(a1 + 200);
    v4 = *(_QWORD *)(a1 + 208);
    v5 = *(_QWORD *)(v3 + 176);
    if ( !*(_BYTE *)(v5 + 280) )
    {
      sub_FF9360(*(_QWORD **)(v3 + 176), *(_QWORD *)(v5 + 288), *(_QWORD *)(v5 + 296), *(__int64 **)(v5 + 304), 0, 0);
      *(_BYTE *)(v5 + 280) = 1;
    }
    sub_FE7D70(v1, *(const char **)(a1 + 192), v5, v4);
    *(_BYTE *)(a1 + 184) = 1;
  }
  return sub_FDC540(v1);
}
