// Function: sub_899080
// Address: 0x899080
//
_QWORD *__fastcall sub_899080(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rbx
  _QWORD *i; // rbx
  __int64 v6; // rdx
  __int64 v7; // rdi

  result = sub_727DC0();
  v3 = a1[9];
  result[1] = a2;
  *result = v3;
  v4 = (_QWORD *)a1[21];
  for ( a1[9] = result; v4; v4 = (_QWORD *)*v4 )
    result = sub_5EDDD0(*(_QWORD *)(v4[3] + 88LL), a2);
  for ( i = (_QWORD *)a1[12]; i; i = (_QWORD *)*i )
  {
    v6 = i[1];
    switch ( *(_BYTE *)(v6 + 80) )
    {
      case 4:
      case 5:
        v7 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 80LL);
        break;
      case 6:
        v7 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v7 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v7 = *(_QWORD *)(v6 + 88);
        break;
      default:
        v7 = 0;
        break;
    }
    result = (_QWORD *)sub_899080(v7, a2);
  }
  return result;
}
