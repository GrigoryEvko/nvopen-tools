// Function: sub_32060E0
// Address: 0x32060e0
//
__int64 __fastcall sub_32060E0(__int64 a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  __int16 v4; // ax
  __int16 v5; // bx
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned int v11; // ebx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v16; // [rsp+20h] [rbp-70h] BYREF
  _DWORD v17[4]; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v18; // [rsp+40h] [rbp-50h]
  unsigned __int64 v19; // [rsp+48h] [rbp-48h]
  __int64 v20; // [rsp+50h] [rbp-40h]
  __int64 v21; // [rsp+58h] [rbp-38h]
  __int64 v22; // [rsp+60h] [rbp-30h]

  if ( (unsigned __int8)sub_31F7430((__int64)a2) )
  {
    result = 3;
    if ( a2 )
      return sub_3205010(a1, (__int64)a2);
  }
  else
  {
    v4 = sub_31F58C0((__int64)a2);
    LOBYTE(v4) = v4 | 0x80;
    v5 = v4;
    sub_3205740((__int64)v15, a1, a2);
    v6 = *(a2 - 16);
    if ( (v6 & 2) != 0 )
      v7 = *((_QWORD *)a2 - 4);
    else
      v7 = (__int64)&a2[-8 * ((v6 >> 2) & 0xF) - 16];
    v8 = *(_QWORD *)(v7 + 56);
    if ( v8 )
      v8 = sub_B91420(v8);
    else
      v9 = 0;
    v20 = v8;
    v17[0] = 5382;
    v18 = v15[0];
    LOWORD(v17[1]) = v5;
    *(_DWORD *)((char *)&v17[1] + 2) = 0;
    v19 = v15[1];
    v21 = v9;
    v22 = 0;
    v10 = sub_370A1A0(a1 + 648, v17);
    v11 = sub_3707F80(a1 + 632, v10);
    if ( (a2[20] & 4) == 0 )
    {
      v14 = *(unsigned int *)(a1 + 1288);
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1292) )
      {
        sub_C8D5F0(a1 + 1280, (const void *)(a1 + 1296), v14 + 1, 8u, v12, v13);
        v14 = *(unsigned int *)(a1 + 1288);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 1280) + 8 * v14) = a2;
      ++*(_DWORD *)(a1 + 1288);
    }
    if ( (__int64 *)v15[0] != &v16 )
      j_j___libc_free_0(v15[0]);
    return v11;
  }
  return result;
}
