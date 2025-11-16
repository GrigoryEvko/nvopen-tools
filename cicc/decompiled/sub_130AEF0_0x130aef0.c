// Function: sub_130AEF0
// Address: 0x130aef0
//
__int64 __fastcall sub_130AEF0(__int64 a1, _OWORD *a2)
{
  __int64 result; // rax

  *a2 = 0;
  a2[1] = 0;
  a2[2] = 0;
  a2[3] = 0;
  sub_130B140((char *)a2 + 8, &unk_42858C8);
  result = sub_130B140(a2, &unk_42858C8);
  *((_QWORD *)a2 + 6) = 0;
  return result;
}
