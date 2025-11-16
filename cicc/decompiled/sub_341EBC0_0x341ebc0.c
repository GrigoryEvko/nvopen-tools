// Function: sub_341EBC0
// Address: 0x341ebc0
//
_QWORD *__fastcall sub_341EBC0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 (*v4)(); // rax
  char v5; // al
  __int64 (*v7)(); // rax
  __int64 (__fastcall *v8)(__int64, _QWORD); // rax

  v2 = *(_QWORD *)(a1 + 808);
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 16LL);
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 184LL);
  if ( v4 != sub_3059410 )
  {
    v8 = (__int64 (__fastcall *)(__int64, _QWORD))((__int64 (__fastcall *)(__int64))v4)(v3);
    if ( v8 )
      return (_QWORD *)v8(a1, a2);
  }
  if ( !a2 )
    return (_QWORD *)sub_33553F0(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v3 + 256LL))(v3) )
  {
    v7 = *(__int64 (**)())(*(_QWORD *)v3 + 264LL);
    if ( v7 == sub_3059450 || ((unsigned __int8 (__fastcall *)(__int64))v7)(v3) )
      return (_QWORD *)sub_33553F0(a1);
  }
  v5 = *(_BYTE *)(v2 + 72);
  switch ( v5 )
  {
    case 1:
      return (_QWORD *)sub_33553F0(a1);
    case 2:
      return (_QWORD *)sub_3355700(a1);
    case 3:
      return (_QWORD *)sub_3354FA0(a1);
    case 5:
      return sub_3363950(a1);
    case 6:
      return (_QWORD *)sub_334CB40(a1);
    case 7:
      return (_QWORD *)sub_334D020(a1);
  }
  return (_QWORD *)sub_3355A10(a1);
}
