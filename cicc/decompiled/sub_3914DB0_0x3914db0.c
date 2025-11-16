// Function: sub_3914DB0
// Address: 0x3914db0
//
_QWORD *__fastcall sub_3914DB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  char *v3; // rcx
  _QWORD *result; // rax
  _QWORD *v5; // rdx
  _QWORD v6[3]; // [rsp+0h] [rbp-20h] BYREF
  char v7; // [rsp+18h] [rbp-8h] BYREF

  v2 = a1 + 168;
  v6[0] = a1 + 168;
  v3 = (char *)v6;
  v6[1] = a1 + 192;
  v6[2] = a1 + 216;
  while ( 1 )
  {
    result = *(_QWORD **)v2;
    v5 = *(_QWORD **)(v2 + 8);
    if ( v5 != result )
      break;
LABEL_7:
    v3 += 8;
    if ( v3 == &v7 )
      return 0;
    v2 = *(_QWORD *)v3;
  }
  while ( *result != a2 )
  {
    result += 3;
    if ( v5 == result )
      goto LABEL_7;
  }
  return result;
}
