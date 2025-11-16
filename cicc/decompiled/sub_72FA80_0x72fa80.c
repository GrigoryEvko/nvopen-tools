// Function: sub_72FA80
// Address: 0x72fa80
//
__int64 __fastcall sub_72FA80(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  char v3; // r13
  unsigned __int8 v4; // [rsp+7h] [rbp-29h] BYREF
  _QWORD *v5; // [rsp+8h] [rbp-28h] BYREF

  if ( (*(_BYTE *)(a1 + 170) & 0x20) != 0 || (result = *(_QWORD *)(a1 + 48)) != 0 && (*(_BYTE *)(result + 195) & 8) != 0 )
  {
    v3 = *(_BYTE *)(a1 + 177);
    sub_72F9F0(a1, 0, &v4, &v5);
    result = v4;
    *(_BYTE *)(a1 + 177) = 0;
    switch ( (_BYTE)result )
    {
      case 1:
        result = sub_76D560(*v5, a2);
        break;
      case 2:
        result = sub_76D400(*v5, a2);
        break;
      case 5:
        result = sub_76CDC0(*v5);
        break;
    }
    *(_BYTE *)(a1 + 177) = v3;
  }
  return result;
}
