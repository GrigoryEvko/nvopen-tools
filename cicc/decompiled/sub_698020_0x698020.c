// Function: sub_698020
// Address: 0x698020
//
__int64 __fastcall sub_698020(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v7; // bl
  __int64 v8; // rsi
  __int64 result; // rax

  v7 = a2;
  if ( (_BYTE)a2 == 86 )
    v8 = sub_72CBE0(a1, a2, a3, a4, a5, a6);
  else
    v8 = sub_73D4C0(*a1, dword_4F077C4 == 2);
  a1[2] = a3;
  result = sub_73DBF0(v7, v8, a1);
  if ( unk_4D04810 )
    *(_BYTE *)(result + 60) |= 2u;
  return result;
}
