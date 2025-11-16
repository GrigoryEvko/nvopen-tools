// Function: sub_3058830
// Address: 0x3058830
//
unsigned __int64 __fastcall sub_3058830(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rdx
  unsigned __int64 result; // rax

  v2 = a2[24];
  a2[34] += 32LL;
  result = (v2 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2[25] >= result + 32 && v2 )
    a2[24] = result + 32;
  else
    result = sub_9D1E70((__int64)(a2 + 24), 32, 32, 3);
  *(_DWORD *)(result + 8) = 4;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = a1;
  *(_QWORD *)result = &unk_4A2F228;
  return result;
}
