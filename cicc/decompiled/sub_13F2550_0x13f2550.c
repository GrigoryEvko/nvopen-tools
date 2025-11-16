// Function: sub_13F2550
// Address: 0x13f2550
//
__int64 __fastcall sub_13F2550(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // r15d
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rbx
  unsigned int v14; // eax
  unsigned int v15; // eax
  bool v16; // zf
  int v19; // [rsp+20h] [rbp-60h] BYREF
  __int64 v20; // [rsp+28h] [rbp-58h] BYREF
  unsigned int v21; // [rsp+30h] [rbp-50h]
  __int64 v22; // [rsp+38h] [rbp-48h] BYREF
  unsigned int v23; // [rsp+40h] [rbp-40h]

  v8 = *(_DWORD *)(*(_QWORD *)a3 + 8LL);
  v9 = sub_157EB90(a4);
  v10 = v8 >> 8;
  v11 = sub_1632FA0(v9);
  v12 = sub_13E7A30(a2 + 4, *a2, v11, a2[3]);
  v19 = 0;
  if ( !(unsigned __int8)sub_13EFC20(v12, a3, a4, a5, &v19, a6) )
  {
    sub_13EFEC0(v12);
    sub_13EFC20(v12, a3, a4, a5, &v19, a6);
  }
  if ( v19 )
  {
    if ( v19 == 3 )
    {
      v14 = v21;
      *(_DWORD *)(a1 + 8) = v21;
      if ( v14 > 0x40 )
        sub_16A4FD0(a1, &v20);
      else
        *(_QWORD *)a1 = v20;
      v15 = v23;
      *(_DWORD *)(a1 + 24) = v23;
      if ( v15 <= 0x40 )
      {
        v16 = v19 == 3;
        *(_QWORD *)(a1 + 16) = v22;
        if ( !v16 )
          return a1;
        goto LABEL_12;
      }
      sub_16A4FD0(a1 + 16, &v22);
    }
    else
    {
      sub_15897D0(a1, v10, 1);
    }
    if ( v19 != 3 )
      return a1;
  }
  else
  {
    sub_15897D0(a1, v10, 0);
    if ( v19 != 3 )
      return a1;
  }
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
LABEL_12:
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  return a1;
}
