// Function: sub_7333B0
// Address: 0x7333b0
//
_BYTE *__fastcall sub_7333B0(__int64 a1, _BYTE *a2, char a3, __int64 a4, __int64 a5)
{
  _BYTE *v8; // rbx
  _BYTE *result; // rax

  v8 = a2;
  if ( !a2 )
    v8 = sub_732EF0(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C);
  result = sub_725B10();
  *(_QWORD *)result = *((_QWORD *)v8 + 25);
  *((_QWORD *)v8 + 25) = result;
  *((_QWORD *)result + 1) = a1;
  *(_BYTE *)(a1 + 177) = 4;
  result[16] = a3;
  if ( a3 == 1 )
  {
    *((_QWORD *)result + 3) = a4;
  }
  else
  {
    if ( a3 != 2 )
      sub_721090();
    *((_QWORD *)result + 3) = a5;
  }
  return result;
}
