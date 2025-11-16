// Function: sub_C4C950
// Address: 0xc4c950
//
__int64 __fastcall sub_C4C950(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  unsigned int v5; // ebx
  unsigned int v6; // eax
  const void *v7; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-48h]
  __int64 v9; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-38h]
  const void *v11; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-28h]

  if ( a4 <= 1 )
  {
    if ( a4 >= 0 )
    {
      sub_C4A1D0(a1, a2, a3);
      return a1;
    }
LABEL_20:
    BUG();
  }
  if ( a4 != 2 )
    goto LABEL_20;
  v8 = 1;
  v7 = 0;
  v10 = 1;
  v9 = 0;
  sub_C4BFE0(a2, a3, &v7, &v9);
  v5 = v10;
  if ( v10 <= 0x40 )
  {
    if ( !v9 )
      goto LABEL_8;
LABEL_16:
    v12 = v8;
    if ( v8 > 0x40 )
      sub_C43780((__int64)&v11, &v7);
    else
      v11 = v7;
    sub_C46A40((__int64)&v11, 1);
    v5 = v10;
    *(_DWORD *)(a1 + 8) = v12;
    *(_QWORD *)a1 = v11;
    goto LABEL_9;
  }
  if ( v5 != (unsigned int)sub_C444A0((__int64)&v9) )
    goto LABEL_16;
LABEL_8:
  v6 = v8;
  v8 = 0;
  *(_DWORD *)(a1 + 8) = v6;
  *(_QWORD *)a1 = v7;
LABEL_9:
  if ( v5 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( v8 <= 0x40 || !v7 )
    return a1;
  j_j___libc_free_0_0(v7);
  return a1;
}
