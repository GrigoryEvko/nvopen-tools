// Function: sub_1368BE0
// Address: 0x1368be0
//
__int64 __fastcall sub_1368BE0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rax

  v3 = *a2;
  if ( *a2 )
  {
    v5 = sub_1368BD0(a2);
    sub_1370CF0(a1, v3, v5, a3);
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 0;
  }
  return a1;
}
