// Function: sub_303BA80
// Address: 0x303ba80
//
__int64 __fastcall sub_303BA80(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r15
  int v11; // r9d
  __int128 v12; // rax
  int v13; // r9d
  __int128 v14; // [rsp-20h] [rbp-70h]
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  int v17; // [rsp+18h] [rbp-38h]

  if ( *(_WORD *)(*(_QWORD *)(a2 + 48) + 16LL * a3) != 10 )
    return a2;
  v6 = *(_QWORD *)(a2 + 80);
  v16 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v16, v6, 1);
  v17 = *(_DWORD *)(a2 + 72);
  v8 = sub_3400D50(a4, 0, &v16, 1);
  v10 = v9;
  *(_QWORD *)&v12 = sub_33FAF80(
                      a4,
                      *(_DWORD *)(a2 + 24),
                      (unsigned int)&v16,
                      12,
                      0,
                      v11,
                      *(_OWORD *)*(_QWORD *)(a2 + 40));
  *((_QWORD *)&v14 + 1) = v10;
  *(_QWORD *)&v14 = v8;
  result = sub_3406EB0(a4, 230, (unsigned int)&v16, 10, 0, v13, v12, v14);
  if ( v16 )
  {
    v15 = result;
    sub_B91220((__int64)&v16, v16);
    return v15;
  }
  return result;
}
