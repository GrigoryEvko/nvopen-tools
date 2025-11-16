// Function: sub_161E760
// Address: 0x161e760
//
unsigned __int64 __fastcall sub_161E760(unsigned __int8 *a1)
{
  int v1; // eax
  unsigned __int64 result; // rax
  __int64 v3; // rax
  unsigned __int64 v4; // rdi
  bool v5; // zf
  unsigned __int8 *v6; // rdi
  bool v7; // cf

  v1 = *a1;
  if ( (unsigned __int8)(v1 - 4) > 0x1Eu )
  {
    v6 = a1 + 8;
    v7 = (unsigned int)(v1 - 1) < 2;
    result = 0;
    if ( v7 )
      return (unsigned __int64)v6;
  }
  else if ( a1[1] == 2 || (result = 0, *((_DWORD *)a1 + 3)) )
  {
    v3 = *((_QWORD *)a1 + 2);
    v4 = v3 & 0xFFFFFFFFFFFFFFF8LL;
    v5 = (v3 & 4) == 0;
    result = 0;
    if ( !v5 )
      return v4;
  }
  return result;
}
