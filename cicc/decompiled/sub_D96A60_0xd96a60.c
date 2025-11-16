// Function: sub_D96A60
// Address: 0xd96a60
//
__int64 __fastcall sub_D96A60(__int64 a1, __int64 a2, __int64 a3, __int16 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  unsigned int v10; // eax
  __int16 v11; // ax
  __int16 v13; // [rsp+4h] [rbp-6Ch]
  __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-58h]
  __int64 v17; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-48h]
  __int64 v19; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-38h]

  v9 = *(unsigned __int16 *)(a5 + 26);
  v16 = 16;
  v15 = 1;
  v20 = 16;
  v19 = v9;
  sub_C49B30((__int64)&v17, (__int64)&v15, &v19);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  v15 = v17;
  v10 = v18;
  v16 = v18;
  if ( v20 > 0x40 && v19 )
  {
    j_j___libc_free_0_0(v19);
    v10 = v16;
  }
  if ( v10 <= 0x40 )
  {
    v11 = v15;
  }
  else
  {
    v13 = *(_WORD *)v15;
    j_j___libc_free_0_0(v15);
    v11 = v13;
  }
  *(_WORD *)(a1 + 26) = v11;
  *(_WORD *)(a1 + 28) = 0;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  *(_WORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 32) = a5;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 40) = a6;
  return a6;
}
