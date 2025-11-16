// Function: sub_5C9870
// Address: 0x5c9870
//
__int64 __fastcall sub_5C9870(_QWORD *a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // r12
  __int64 result; // rax

  v1 = (_QWORD *)sub_877FE0(*a1);
  v2 = sub_736C60(76, v1[13]);
  if ( (unsigned int)sub_8D2600(v1[20]) )
  {
    result = sub_684B30(1651, v2 + 56);
    *(_BYTE *)(v2 + 8) = 0;
  }
  else
  {
    result = v1[21];
    *(_BYTE *)(result + 20) |= 4u;
  }
  return result;
}
