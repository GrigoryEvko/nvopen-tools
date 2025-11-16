// Function: sub_17A2480
// Address: 0x17a2480
//
unsigned __int64 __fastcall sub_17A2480(_DWORD *a1, __int64 a2)
{
  unsigned int v2; // ecx
  unsigned __int64 result; // rax

  v2 = a1[2];
  if ( v2 > 0x40 )
  {
    **(_QWORD **)a1 = a2;
    return (unsigned __int64)memset(
                               (void *)(*(_QWORD *)a1 + 8LL),
                               0,
                               8 * (unsigned int)(((unsigned __int64)(unsigned int)a1[2] + 63) >> 6) - 8);
  }
  else
  {
    result = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    *(_QWORD *)a1 = result & a2;
  }
  return result;
}
