// Function: sub_AD5CE0
// Address: 0xad5ce0
//
__int64 __fastcall sub_AD5CE0(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 **a5)
{
  __int64 v7; // r12
  __int64 v9; // rax
  __int64 **v10; // rax
  __int64 v11; // rdi
  __int64 v13; // [rsp+18h] [rbp-A8h]
  _QWORD v14[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v15[6]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v16; // [rsp+60h] [rbp-60h]
  unsigned int v17; // [rsp+68h] [rbp-58h]
  __int64 v18; // [rsp+70h] [rbp-50h]
  unsigned int v19; // [rsp+78h] [rbp-48h]
  char v20; // [rsp+80h] [rbp-40h]

  v7 = sub_AAA8B0(a1, a2, a3, a4);
  if ( !v7 )
  {
    v9 = *(_QWORD *)(a1 + 8);
    LODWORD(v13) = a4;
    BYTE4(v13) = *(_BYTE *)(v9 + 8) == 18;
    v10 = (__int64 **)sub_BCE1B0(*(_QWORD *)(v9 + 24), v13);
    if ( v10 != a5 )
    {
      v14[0] = a1;
      v14[1] = a2;
      v11 = **v10;
      v15[2] = 2;
      v15[0] = 63;
      v15[1] = v14;
      v15[3] = a3;
      v15[4] = a4;
      v15[5] = 0;
      v20 = 0;
      v7 = sub_AD4210(v11 + 2120, (__int64)v10, (__int16 *)v15);
      if ( v20 )
      {
        v20 = 0;
        if ( v19 > 0x40 && v18 )
          j_j___libc_free_0_0(v18);
        if ( v17 > 0x40 && v16 )
          j_j___libc_free_0_0(v16);
      }
    }
  }
  return v7;
}
