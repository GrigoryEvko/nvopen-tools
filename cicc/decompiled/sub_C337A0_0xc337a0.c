// Function: sub_C337A0
// Address: 0xc337a0
//
__int64 __fastcall sub_C337A0(__int64 a1)
{
  unsigned int v1; // ecx
  unsigned int v2; // eax
  unsigned int v3; // eax
  unsigned int v4; // edx
  __int64 result; // rax

  v1 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
  v2 = (v1 + 63) >> 6;
  if ( !v2 )
    v2 = 1;
  v3 = v2 << 6;
  v4 = v3 + 1 - v1;
  result = v3 - v1;
  if ( v1 >= 2 )
    return v4;
  return result;
}
