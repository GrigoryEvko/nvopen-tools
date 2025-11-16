// Function: sub_2FE3E80
// Address: 0x2fe3e80
//
bool __fastcall sub_2FE3E80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  __int64 (*v11)(); // rax
  bool result; // al
  __int64 v14; // rdi
  unsigned int v15; // r12d
  __int64 v16; // rdi
  unsigned int v17; // r12d
  bool v18; // dl

  v11 = *(__int64 (**)())(*(_QWORD *)a1 + 400LL);
  if ( v11 == sub_2FE3030
    || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v11)(a1, a2, a3, a7, a8) )
  {
    return a4 == 0;
  }
  if ( a6 != 190
    || ((v16 = *(_QWORD *)(a5 + 96), v17 = *(_DWORD *)(v16 + 32), v17 <= 0x40)
      ? (v18 = *(_QWORD *)(v16 + 24) == 1)
      : (v18 = v17 - 1 == (unsigned int)sub_C444A0(v16 + 24)),
        result = 0,
        !v18) )
  {
    if ( !a4 || a9 != 190 )
      return a4 == 0;
    v14 = *(_QWORD *)(a4 + 96);
    v15 = *(_DWORD *)(v14 + 32);
    result = v15 <= 0x40 ? *(_QWORD *)(v14 + 24) == 1 : v15 - 1 == (unsigned int)sub_C444A0(v14 + 24);
    if ( !result )
      return a4 == 0;
  }
  return result;
}
