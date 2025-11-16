// Function: sub_144B4E0
// Address: 0x144b4e0
//
__int64 __fastcall sub_144B4E0(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // rax
  __int64 result; // rax
  __int64 *v4; // r13
  __int64 *i; // rbx
  __int64 v6; // rdi
  __int64 v7; // r13
  _QWORD *v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rdx

  v2 = (_QWORD *)a2[6];
  if ( v2 == (_QWORD *)(a2[8] - 8) )
  {
    v7 = a2[9];
    if ( (((__int64)v2 - a2[7]) >> 3) + ((((v7 - a2[5]) >> 3) - 1) << 6) + ((a2[4] - a2[2]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( (unsigned __int64)(a2[1] - ((v7 - *a2) >> 3)) <= 1 )
    {
      sub_144B360(a2, 1u, 0);
      v7 = a2[9];
    }
    *(_QWORD *)(v7 + 8) = sub_22077B0(512);
    v8 = (_QWORD *)a2[6];
    if ( v8 )
      *v8 = a1;
    v9 = (__int64 *)(a2[9] + 8);
    a2[9] = (__int64)v9;
    result = *v9;
    v10 = *v9 + 512;
    a2[7] = result;
    a2[8] = v10;
    a2[6] = result;
  }
  else
  {
    if ( v2 )
    {
      *v2 = a1;
      v2 = (_QWORD *)a2[6];
    }
    result = (__int64)(v2 + 1);
    a2[6] = result;
  }
  v4 = *(__int64 **)(a1 + 48);
  for ( i = *(__int64 **)(a1 + 40); v4 != i; result = sub_144B4E0(v6, a2) )
    v6 = *i++;
  return result;
}
