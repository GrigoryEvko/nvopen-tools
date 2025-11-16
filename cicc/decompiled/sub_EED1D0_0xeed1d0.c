// Function: sub_EED1D0
// Address: 0xeed1d0
//
__int64 __fastcall sub_EED1D0(_QWORD *a1)
{
  __int64 v1; // rbp
  _BYTE *v2; // rax
  _BYTE *v3; // rdx
  char v5; // dl
  _BYTE *v6; // rax
  __int64 v7[2]; // [rsp-10h] [rbp-10h] BYREF

  v2 = (_BYTE *)*a1;
  v3 = (_BYTE *)a1[1];
  if ( (_BYTE *)*a1 == v3 )
    return 0;
  if ( *v2 != 68 )
    return 0;
  *a1 = v2 + 1;
  if ( v3 == v2 + 1 )
    return 0;
  v5 = v2[1];
  if ( v5 != 116 && v5 != 84 )
    return 0;
  *a1 = v2 + 2;
  v7[1] = v1;
  v7[0] = sub_EEA9F0((__int64)a1);
  if ( !v7[0] )
    return 0;
  v6 = (_BYTE *)*a1;
  if ( *a1 == a1[1] || *v6 != 69 )
    return 0;
  *a1 = v6 + 1;
  return sub_EE7130((__int64)(a1 + 101), "decltype", v7);
}
