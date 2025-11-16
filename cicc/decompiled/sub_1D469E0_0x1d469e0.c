// Function: sub_1D469E0
// Address: 0x1d469e0
//
_QWORD *__fastcall sub_1D469E0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 (*v4)(); // rax
  int v5; // eax
  __int64 (*v7)(); // rax
  __int64 (__fastcall *v8)(__int64, _QWORD); // rax

  v2 = *(_QWORD *)(a1 + 320);
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 256) + 16LL);
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 96LL);
  if ( v4 != sub_1D45F90 )
  {
    v8 = (__int64 (__fastcall *)(__int64, _QWORD))((__int64 (__fastcall *)(__int64))v4)(v3);
    if ( v8 )
      return (_QWORD *)v8(a1, a2);
  }
  if ( !a2 )
    return (_QWORD *)sub_1D05510(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v3 + 144LL))(v3) )
  {
    v7 = *(__int64 (**)())(*(_QWORD *)v3 + 160LL);
    if ( v7 == sub_1D45FA0 || ((unsigned __int8 (__fastcall *)(__int64))v7)(v3) )
      return (_QWORD *)sub_1D05510(a1);
  }
  v5 = *(_DWORD *)(v2 + 72);
  switch ( v5 )
  {
    case 1:
      return (_QWORD *)sub_1D05510(a1);
    case 2:
      return (_QWORD *)sub_1D05200(a1);
    case 3:
      return (_QWORD *)sub_1D05820(a1);
    case 5:
      return sub_1D122D0(a1);
  }
  return (_QWORD *)sub_1D04DC0(a1);
}
