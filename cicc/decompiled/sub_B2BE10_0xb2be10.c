// Function: sub_B2BE10
// Address: 0xb2be10
//
bool __fastcall sub_B2BE10(__int64 a1)
{
  __int64 v1; // rbp
  __int64 v3[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) != 14 )
    return 0;
  v3[1] = v1;
  v3[0] = sub_B2BDE0(a1);
  return (unsigned __int16)sub_A73B10(v3) == 0;
}
