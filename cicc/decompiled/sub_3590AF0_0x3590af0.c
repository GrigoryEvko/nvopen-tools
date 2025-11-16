// Function: sub_3590AF0
// Address: 0x3590af0
//
__int64 __fastcall sub_3590AF0(__int64 *a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 v8; // r9
  __int64 v9; // rsi
  unsigned __int64 v10; // rsi
  unsigned int v11; // r12d
  __int64 v13; // r9
  bool v14; // dl
  bool v15; // al
  unsigned __int8 v16; // r10
  _BYTE *v17; // rsi
  _QWORD v18[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v19; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int64 v20; // [rsp+18h] [rbp-28h]
  __int64 v21; // [rsp+20h] [rbp-20h] BYREF

  v18[0] = a3;
  v18[1] = a4;
  sub_C93130((__int64 *)&v19, (__int64)v18);
  v6 = *a1;
  v7 = a2 & 0x7FFFFFFF;
  v8 = a2 & 0x7FFFFFFF;
  v9 = *(_QWORD *)(*(_QWORD *)(*a1 + 56) + 16 * v8);
  if ( v9 && (v9 & 4) == 0 && (v10 = v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v11 = sub_2EC06C0(v6, v10, v19, v20, (__int64)v19, v8);
  }
  else
  {
    if ( a2 >= 0 || v7 >= *(_DWORD *)(v6 + 464) )
    {
      v13 = 0;
      v14 = 0;
      v15 = 0;
      v16 = 0;
    }
    else
    {
      v17 = (_BYTE *)(*(_QWORD *)(v6 + 456) + 8 * v8);
      v16 = *v17 & 1;
      v14 = (*v17 & 4) != 0;
      v13 = *(_QWORD *)v17 >> 3;
      v15 = (*v17 & 2) != 0;
    }
    v11 = sub_2EC0910(v6, (8 * v13) | (4LL * v14) | v16 | (2LL * v15), v19, v20, (__int64)v19, v13);
  }
  if ( v19 != &v21 )
    j_j___libc_free_0((unsigned __int64)v19);
  return v11;
}
