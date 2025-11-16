// Function: sub_2ECA220
// Address: 0x2eca220
//
__int64 __fastcall sub_2ECA220(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 *v4; // r12
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rsi
  bool v10; // zf
  __int64 v11; // rdx
  __int64 *i; // [rsp+8h] [rbp-68h]
  __int64 v13; // [rsp+10h] [rbp-60h] BYREF
  int v14; // [rsp+18h] [rbp-58h]
  __int64 v15; // [rsp+20h] [rbp-50h]
  __int128 v16; // [rsp+28h] [rbp-48h]
  __int64 v17; // [rsp+38h] [rbp-38h]

  result = *(_QWORD *)(a2 + 72);
  v4 = *(__int64 **)(a2 + 64);
  for ( i = (__int64 *)result; i != v4; ++v4 )
  {
    v7 = *a3;
    v8 = *v4;
    v16 = 0u;
    v9 = a1[17];
    v10 = *(_DWORD *)(a2 + 24) == 1;
    v13 = v7;
    LODWORD(v7) = *((_DWORD *)a3 + 2);
    v15 = v8;
    v14 = v7;
    v11 = a1[2];
    BYTE1(v16) = v10;
    v17 = 0;
    sub_2EC8FB0((__int64)&v13, v9, v11);
    result = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, __int64 *))(*a1 + 144LL))(a1, a3, &v13);
    if ( (_BYTE)result )
    {
      a3[2] = v15;
      *((_WORD *)a3 + 12) = v16;
      *(__int64 *)((char *)a3 + 26) = *(_QWORD *)((char *)&v16 + 2);
      *(_DWORD *)((char *)a3 + 34) = *(_DWORD *)((char *)&v16 + 10);
      result = v17;
      a3[5] = v17;
    }
  }
  return result;
}
