// Function: sub_D15AE0
// Address: 0xd15ae0
//
__int64 __fastcall sub_D15AE0(__int64 a1, int a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  unsigned int v7; // eax
  unsigned int v8; // eax
  __int64 v10; // rdx
  __int64 v11; // [rsp+8h] [rbp-58h]
  __int64 v12; // [rsp+8h] [rbp-58h]
  __int64 v13; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-38h]

  v7 = *(_DWORD *)(a5 + 24);
  v14 = 1;
  v13 = 0;
  v16 = 1;
  v15 = 0;
  if ( v7 > 0x40 )
  {
    v11 = a4;
    sub_C43990((__int64)&v13, a5 + 16);
    a4 = v11;
    if ( v16 > 0x40 )
      goto LABEL_4;
    v8 = *(_DWORD *)(a5 + 8);
    if ( v8 > 0x40 )
      goto LABEL_4;
LABEL_12:
    v10 = *(_QWORD *)a5;
    v16 = v8;
    v15 = v10;
    goto LABEL_5;
  }
  v14 = v7;
  v8 = *(_DWORD *)(a5 + 8);
  v13 = *(_QWORD *)(a5 + 16);
  if ( v8 <= 0x40 )
    goto LABEL_12;
LABEL_4:
  v12 = a4;
  sub_C43990((__int64)&v15, a5);
  a4 = v12;
LABEL_5:
  sub_D14B20(a1, a2, a3, a4, (__int64)&v13, 0, 1u);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  return a1;
}
