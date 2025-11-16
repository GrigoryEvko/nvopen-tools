// Function: sub_2F9A6D0
// Address: 0x2f9a6d0
//
__int64 __fastcall sub_2F9A6D0(__int64 a1, _BYTE *a2)
{
  unsigned int v2; // r12d
  unsigned __int64 v5; // rdi
  unsigned int v6; // r13d
  unsigned __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *a2 == 86 && (v2 = sub_BC8C50((__int64)a2, &v7, v8), (_BYTE)v2) && (v5 = v7, v8[0] + v7) )
  {
    if ( v7 < v8[0] )
      v5 = v8[0];
    v6 = sub_F02DD0(v5, v8[0] + v7);
    if ( (unsigned int)sub_DF95A0(*(_QWORD **)(a1 + 24)) >= v6 )
      return 0;
  }
  else
  {
    return 0;
  }
  return v2;
}
