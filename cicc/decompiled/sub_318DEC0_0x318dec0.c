// Function: sub_318DEC0
// Address: 0x318dec0
//
unsigned __int64 __fastcall sub_318DEC0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 result; // rax

  *a1 = &unk_4A346F0;
  a1[1] = a2;
  a1[2] = 0;
  v2 = sub_318B4B0(a2);
  if ( v2 )
    result = v2 & 0xFFFFFFFFFFFFFFFBLL;
  else
    result = sub_318B4F0(a2) | 4;
  a1[2] = result;
  return result;
}
