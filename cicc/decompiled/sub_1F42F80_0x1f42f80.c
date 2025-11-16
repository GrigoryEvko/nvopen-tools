// Function: sub_1F42F80
// Address: 0x1f42f80
//
__int64 __fastcall sub_1F42F80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  int v11; // eax
  __int64 v12; // rbx
  __int64 v13; // r15
  int v14; // r12d
  __int64 v15; // r15
  char v16; // al
  unsigned int v17; // eax
  char v18; // [rsp+1Bh] [rbp-85h] BYREF
  unsigned int v19; // [rsp+1Ch] [rbp-84h] BYREF
  __int64 v20; // [rsp+20h] [rbp-80h] BYREF
  __int64 v21; // [rsp+28h] [rbp-78h]
  _QWORD v22[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v23; // [rsp+40h] [rbp-60h] BYREF
  __int64 v24; // [rsp+48h] [rbp-58h]
  char v25[8]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v26; // [rsp+58h] [rbp-48h]
  __int64 v27; // [rsp+60h] [rbp-40h]

  v20 = a4;
  v21 = a5;
  if ( (_BYTE)a4 )
    return *(unsigned __int8 *)(a1 + (unsigned __int8)a4 + 1040);
  if ( (unsigned __int8)sub_1F58D20(&v20) )
  {
    v25[0] = 0;
    v26 = 0;
    LOBYTE(v22[0]) = 0;
    return sub_1F426C0(a1, a2, (unsigned int)v20, a5, (__int64)v25, (unsigned int *)&v23, v22);
  }
  else
  {
    v11 = sub_1F58D40(&v20, a2, v7, v8, v9, v10);
    v12 = v20;
    v13 = v21;
    v14 = v11;
    v22[0] = v20;
    v22[1] = v21;
    if ( (unsigned __int8)sub_1F58D20(v22) )
    {
      v25[0] = 0;
      v26 = 0;
      LOBYTE(v19) = 0;
      sub_1F426C0(a1, a2, LODWORD(v22[0]), v13, (__int64)v25, (unsigned int *)&v23, &v19);
      v16 = v19;
    }
    else
    {
      sub_1F40D10((__int64)v25, a1, a2, v12, v13);
      v15 = v27;
      LOBYTE(v23) = v26;
      v24 = v27;
      if ( (_BYTE)v26 )
      {
        v16 = *(_BYTE *)(a1 + (unsigned __int8)v26 + 1155);
      }
      else if ( (unsigned __int8)sub_1F58D20(&v23) )
      {
        v25[0] = 0;
        v26 = 0;
        v18 = 0;
        sub_1F426C0(a1, a2, (unsigned int)v23, v15, (__int64)v25, &v19, &v18);
        v16 = v18;
      }
      else
      {
        sub_1F40D10((__int64)v25, a1, a2, v23, v24);
        v16 = sub_1D5E9F0(a1, a2, (unsigned __int8)v26, v27);
      }
    }
    v25[0] = v16;
    v17 = sub_1F3E310(v25);
    return (v17 + v14 - 1) / v17;
  }
}
