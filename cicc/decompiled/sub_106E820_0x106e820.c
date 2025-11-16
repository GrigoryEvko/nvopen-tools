// Function: sub_106E820
// Address: 0x106e820
//
__int64 __fastcall sub_106E820(__int64 a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // r12
  _QWORD *v7; // rdi
  __int64 result; // rax
  _QWORD *v9; // r13
  _QWORD *v10; // r12
  _QWORD *v11; // rdi
  __int64 v12; // rdi

  v3 = *(_QWORD *)(a1 + 440);
  *(_QWORD *)a1 = &unk_49E6008;
  if ( *(_DWORD *)(a1 + 452) )
  {
    v4 = *(unsigned int *)(a1 + 448);
    if ( (_DWORD)v4 )
    {
      v5 = 8 * v4;
      v6 = 0;
      do
      {
        v7 = *(_QWORD **)(v3 + v6);
        if ( v7 != (_QWORD *)-8LL && v7 )
        {
          a2 = *v7 + 17LL;
          sub_C7D6A0((__int64)v7, a2, 8);
          v3 = *(_QWORD *)(a1 + 440);
        }
        v6 += 8;
      }
      while ( v5 != v6 );
    }
  }
  _libc_free(v3, a2);
  result = sub_C7D6A0(*(_QWORD *)(a1 + 416), 24LL * *(unsigned int *)(a1 + 432), 8);
  v9 = *(_QWORD **)(a1 + 368);
  v10 = *(_QWORD **)(a1 + 360);
  if ( v9 != v10 )
  {
    do
    {
      v11 = (_QWORD *)v10[9];
      result = (__int64)(v10 + 11);
      if ( v11 != v10 + 11 )
        result = j_j___libc_free_0(v11, v10[11] + 1LL);
      v12 = v10[6];
      if ( v12 )
        result = j_j___libc_free_0(v12, v10[8] - v12);
      v10 += 13;
    }
    while ( v9 != v10 );
    v10 = *(_QWORD **)(a1 + 360);
  }
  if ( v10 )
    return j_j___libc_free_0(v10, *(_QWORD *)(a1 + 376) - (_QWORD)v10);
  return result;
}
