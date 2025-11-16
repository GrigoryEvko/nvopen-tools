// Function: sub_15A3950
// Address: 0x15a3950
//
__int64 __fastcall sub_15A3950(__int64 a1, __int64 a2, _BYTE *a3, __int64 **a4)
{
  __int64 v7; // r12
  __int64 **v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rsi
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int128 v17; // [rsp-30h] [rbp-B0h]
  _QWORD v18[16]; // [rsp+0h] [rbp-80h] BYREF

  v7 = sub_1584900(a1, a2, a3);
  if ( !v7 )
  {
    v9 = (__int64 **)sub_16463B0(**(_QWORD **)(*(_QWORD *)a1 + 16LL), *(_QWORD *)(*(_QWORD *)a3 + 32LL));
    v14 = (__int64)v9;
    if ( a4 != v9 )
    {
      v18[0] = a1;
      v18[1] = a2;
      v18[2] = a3;
      v18[5] = v18;
      v15 = *v9;
      v18[4] = 61;
      v16 = *v15;
      v18[6] = 3;
      memset(&v18[7], 0, 24);
      *((_QWORD *)&v17 + 1) = v18;
      *(_QWORD *)&v17 = 61;
      return sub_15A2780(v16 + 1776, v14, v10, v11, v12, v13, v17, 3u, 0);
    }
  }
  return v7;
}
