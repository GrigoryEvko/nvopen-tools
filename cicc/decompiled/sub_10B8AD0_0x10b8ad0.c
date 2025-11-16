// Function: sub_10B8AD0
// Address: 0x10b8ad0
//
__int64 __fastcall sub_10B8AD0(__int64 a1, _DWORD *a2, __int64 a3, __int64 *a4, __int64 *a5)
{
  unsigned int v8; // r15d
  __int64 v10; // rax
  __int64 v11; // rax
  bool v12; // zf
  __int64 v13; // [rsp+0h] [rbp-70h] BYREF
  int v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-58h]
  __int64 v17; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-48h]
  unsigned __int8 v19; // [rsp+30h] [rbp-40h]

  sub_11FBC10(&v13, a1, 1, 1, a5);
  v8 = v19;
  if ( v19 )
  {
    *a2 = v14;
    v10 = v13;
    *(_QWORD *)a3 = v13;
    *a4 = sub_AD8D80(*(_QWORD *)(v10 + 8), (__int64)&v15);
    v11 = sub_AD8D80(*(_QWORD *)(*(_QWORD *)a3 + 8LL), (__int64)&v17);
    v12 = v19 == 0;
    *a5 = v11;
    if ( !v12 )
    {
      v19 = 0;
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
      if ( v16 > 0x40 && v15 )
        j_j___libc_free_0_0(v15);
    }
  }
  return v8;
}
