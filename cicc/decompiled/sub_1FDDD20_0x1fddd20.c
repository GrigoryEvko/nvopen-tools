// Function: sub_1FDDD20
// Address: 0x1fddd20
//
__int64 __fastcall sub_1FDDD20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // eax
  __int64 v7; // rbx
  __int64 v8; // r15
  int v9; // r12d
  __int64 v10; // r15
  char v11; // di
  unsigned int v12; // eax
  __int64 v13; // [rsp+10h] [rbp-90h] BYREF
  __int64 v14; // [rsp+18h] [rbp-88h]
  char v15; // [rsp+2Bh] [rbp-75h] BYREF
  unsigned int v16; // [rsp+2Ch] [rbp-74h] BYREF
  _QWORD v17[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v18; // [rsp+40h] [rbp-60h] BYREF
  __int64 v19; // [rsp+48h] [rbp-58h]
  char v20[8]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v21; // [rsp+58h] [rbp-48h]
  __int64 v22; // [rsp+60h] [rbp-40h]

  v13 = a3;
  v14 = a4;
  if ( (_BYTE)a3 )
    return *(unsigned __int8 *)(a1 + (unsigned __int8)a3 + 1040);
  if ( sub_1F58D20((__int64)&v13) )
  {
    v20[0] = 0;
    v21 = 0;
    LOBYTE(v17[0]) = 0;
    return sub_1F426C0(a1, a2, (unsigned int)v13, a4, (__int64)v20, (unsigned int *)&v18, v17);
  }
  else
  {
    v6 = sub_1F58D40((__int64)&v13);
    v7 = v13;
    v8 = v14;
    v9 = v6;
    v17[0] = v13;
    v17[1] = v14;
    if ( sub_1F58D20((__int64)v17) )
    {
      v20[0] = 0;
      v21 = 0;
      LOBYTE(v16) = 0;
      sub_1F426C0(a1, a2, LODWORD(v17[0]), v8, (__int64)v20, (unsigned int *)&v18, &v16);
      v11 = v16;
    }
    else
    {
      sub_1F40D10((__int64)v20, a1, a2, v7, v8);
      v10 = v22;
      LOBYTE(v18) = v21;
      v19 = v22;
      if ( (_BYTE)v21 )
      {
        v11 = *(_BYTE *)(a1 + (unsigned __int8)v21 + 1155);
      }
      else if ( sub_1F58D20((__int64)&v18) )
      {
        v20[0] = 0;
        v21 = 0;
        v15 = 0;
        sub_1F426C0(a1, a2, (unsigned int)v18, v10, (__int64)v20, &v16, &v15);
        v11 = v15;
      }
      else
      {
        sub_1F40D10((__int64)v20, a1, a2, v18, v19);
        v11 = sub_1D5E9F0(a1, a2, (unsigned __int8)v21, v22);
      }
    }
    v12 = sub_1FDDC20(v11);
    return (v12 + v9 - 1) / v12;
  }
}
