// Function: sub_89A0D0
// Address: 0x89a0d0
//
__int64 __fastcall sub_89A0D0(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rdx
  __int64 result; // rax

  v1 = sub_725240();
  v2 = qword_4F601D0;
  v1[1] = a1;
  *v1 = v2;
  qword_4F601D0 = (__int64)v1;
  *(_BYTE *)(a1 + 172) |= 0x40u;
  result = (unsigned int)dword_4F601E0;
  if ( dword_4F601E0 )
    return sub_899AF0();
  return result;
}
