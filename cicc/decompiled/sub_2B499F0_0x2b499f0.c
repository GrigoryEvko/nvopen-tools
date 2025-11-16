// Function: sub_2B499F0
// Address: 0x2b499f0
//
unsigned __int64 __fastcall sub_2B499F0(unsigned __int64 *a1, _QWORD *a2, _DWORD *a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rax
  unsigned __int64 *v8; // rdx
  __int64 v9; // rdx

  v4 = a1[6];
  if ( v4 == a1[8] - 16 )
  {
    v6 = a1[9];
    if ( ((__int64)(v4 - a1[7]) >> 4) + 32 * (((__int64)(v6 - a1[5]) >> 3) - 1) + ((__int64)(a1[4] - a1[2]) >> 4) == 0x7FFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( a1[1] - ((__int64)(v6 - *a1) >> 3) <= 1 )
    {
      sub_2B49870(a1, 1u, 0);
      v6 = a1[9];
    }
    *(_QWORD *)(v6 + 8) = sub_22077B0(0x200u);
    v7 = a1[6];
    if ( v7 )
    {
      *(_QWORD *)v7 = *a2;
      *(_DWORD *)(v7 + 8) = *a3;
    }
    v8 = (unsigned __int64 *)(a1[9] + 8);
    a1[9] = (unsigned __int64)v8;
    result = *v8;
    v9 = *v8 + 512;
    a1[7] = result;
    a1[8] = v9;
    a1[6] = result;
  }
  else
  {
    if ( v4 )
    {
      *(_QWORD *)v4 = *a2;
      *(_DWORD *)(v4 + 8) = *a3;
      v4 = a1[6];
    }
    result = v4 + 16;
    a1[6] = result;
  }
  return result;
}
