// Function: sub_388F790
// Address: 0x388f790
//
__int64 __fastcall sub_388F790(__int64 a1, _QWORD *a2, _DWORD *a3)
{
  __int64 result; // rax
  unsigned __int64 v6; // rcx
  __int64 v7; // rsi

  result = sub_388AF10(a1, 371, "expected GV ID");
  if ( !(_BYTE)result )
  {
    v6 = *(unsigned int *)(a1 + 104);
    *a3 = v6;
    v7 = *(_QWORD *)(a1 + 1320);
    if ( v6 >= (*(_QWORD *)(a1 + 1328) - v7) >> 3 )
      *a2 = qword_5052688;
    else
      *a2 = *(_QWORD *)(v7 + 8 * v6);
  }
  return result;
}
