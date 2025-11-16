// Function: sub_89A080
// Address: 0x89a080
//
__int64 __fastcall sub_89A080(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rdx
  __int64 result; // rax

  v1 = sub_725220();
  v2 = qword_4F601D8;
  v1[1] = a1;
  *v1 = v2;
  qword_4F601D8 = (__int64)v1;
  *(_BYTE *)(a1 + 203) |= 0x80u;
  result = (unsigned int)dword_4F601E0;
  if ( dword_4F601E0 )
    return sub_899AF0();
  return result;
}
