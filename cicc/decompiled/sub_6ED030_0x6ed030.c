// Function: sub_6ED030
// Address: 0x6ed030
//
__int64 __fastcall sub_6ED030(__int64 a1)
{
  _BYTE *v1; // rbx
  char v2; // al
  unsigned __int64 v3; // rax
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // rdx

  v1 = *(_BYTE **)(a1 + 144);
  v2 = v1[24];
  if ( v2 == 5
    || v2 == 1 && (v3 = (unsigned __int8)v1[56], (unsigned __int8)v3 <= 0x13u) && (v6 = 524432, _bittest64(&v6, v3)) )
  {
    result = v1[25] & 0xFC | 1u;
    v1[25] = v1[25] & 0xFC | 1;
  }
  else
  {
    v4 = sub_73E250(*(_QWORD *)(a1 + 144));
    result = sub_73DDB0(v4);
    if ( v1 != (_BYTE *)result )
      *(_QWORD *)(a1 + 144) = result;
  }
  return result;
}
