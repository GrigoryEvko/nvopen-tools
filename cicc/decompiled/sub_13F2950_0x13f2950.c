// Function: sub_13F2950
// Address: 0x13f2950
//
__int64 __fastcall sub_13F2950(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned int v12; // eax
  unsigned int v13; // eax
  bool v14; // zf
  unsigned int v15; // [rsp+Ch] [rbp-64h]
  int v16; // [rsp+10h] [rbp-60h] BYREF
  __int64 v17; // [rsp+18h] [rbp-58h] BYREF
  unsigned int v18; // [rsp+20h] [rbp-50h]
  __int64 v19; // [rsp+28h] [rbp-48h] BYREF
  unsigned int v20; // [rsp+30h] [rbp-40h]

  v15 = *(_DWORD *)(*(_QWORD *)a3 + 8LL) >> 8;
  v8 = sub_157EB90(a4);
  v9 = sub_1632FA0(v8);
  v10 = sub_13E7A30(a2 + 4, *a2, v9, a2[3]);
  sub_13F2700(&v16, v10, a3, a4, a5);
  if ( v16 )
  {
    if ( v16 == 3 )
    {
      v12 = v18;
      *(_DWORD *)(a1 + 8) = v18;
      if ( v12 > 0x40 )
        sub_16A4FD0(a1, &v17);
      else
        *(_QWORD *)a1 = v17;
      v13 = v20;
      *(_DWORD *)(a1 + 24) = v20;
      if ( v13 <= 0x40 )
      {
        v14 = v16 == 3;
        *(_QWORD *)(a1 + 16) = v19;
        if ( !v14 )
          return a1;
        goto LABEL_10;
      }
      sub_16A4FD0(a1 + 16, &v19);
    }
    else
    {
      sub_15897D0(a1, v15, 1);
    }
    if ( v16 != 3 )
      return a1;
  }
  else
  {
    sub_15897D0(a1, v15, 0);
    if ( v16 != 3 )
      return a1;
  }
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
LABEL_10:
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  return a1;
}
