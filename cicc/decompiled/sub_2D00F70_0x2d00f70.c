// Function: sub_2D00F70
// Address: 0x2d00f70
//
__int64 __fastcall sub_2D00F70(unsigned int *a1, _BYTE *a2)
{
  _BYTE *v2; // r15
  unsigned int v3; // r14d
  unsigned int v4; // eax
  int v5; // r14d
  int v6; // r8d
  __int64 result; // rax

  v2 = (_BYTE *)*((_QWORD *)a2 - 8);
  v3 = sub_2D00C30(a1, *((_BYTE **)a2 - 4));
  v4 = sub_2D00C30(a1, v2);
  v5 = sub_2D00850((__int64)a1, v4, v3);
  v6 = sub_2D00C30(a1, a2);
  result = 0;
  if ( v6 != v5 )
  {
    sub_2D00AD0(a1, (unsigned __int64)a2, v5);
    return 1;
  }
  return result;
}
