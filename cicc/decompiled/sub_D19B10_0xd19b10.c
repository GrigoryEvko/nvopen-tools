// Function: sub_D19B10
// Address: 0xd19b10
//
__int64 __fastcall sub_D19B10(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r9
  int v12; // edx
  unsigned __int64 v13; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // r9
  unsigned __int64 v20; // rax
  unsigned int v21; // eax
  unsigned int v22; // eax
  bool v23; // cc
  __int64 v24; // [rsp+10h] [rbp-B0h]
  char v26; // [rsp+2Fh] [rbp-91h] BYREF
  __int64 v27; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-88h]
  unsigned __int64 v29; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-78h]
  __int64 v31; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v32; // [rsp+58h] [rbp-68h]
  __int64 v33; // [rsp+60h] [rbp-60h]
  unsigned int v34; // [rsp+68h] [rbp-58h]
  __int64 v35; // [rsp+70h] [rbp-50h] BYREF
  __int64 v36; // [rsp+78h] [rbp-48h]
  __int64 v37; // [rsp+80h] [rbp-40h]
  unsigned int v38; // [rsp+88h] [rbp-38h]

  v5 = a3[3];
  v6 = *(_QWORD *)(*a3 + 8);
  v7 = sub_B43CC0(v5);
  v8 = v6;
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v6 + 16);
  v9 = sub_9208B0(v7, v8);
  v36 = v10;
  v35 = v9;
  v11 = sub_CA1930(&v35);
  v12 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned int)(v12 - 17) <= 1 )
    LOBYTE(v12) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
  if ( (_BYTE)v12 == 12 )
  {
    v24 = v11;
    if ( (unsigned __int8)sub_D19970(a2, a3) )
    {
      *(_DWORD *)(a1 + 8) = v24;
      if ( (unsigned int)v24 > 0x40 )
        sub_C43690(a1, 0, 0);
      else
        *(_QWORD *)a1 = 0;
    }
    else
    {
      sub_D19710(a2, (__int64)a3, v15, v16, v17, v24);
      sub_D19730((__int64)&v27, a2, v5, v18, (__int64)&v27, v19);
      v30 = v24;
      if ( (unsigned int)v24 > 0x40 )
      {
        sub_C43690((__int64)&v29, -1, 1);
      }
      else
      {
        v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v24;
        if ( !(_DWORD)v24 )
          v20 = 0;
        v29 = v20;
      }
      v32 = 1;
      v31 = 0;
      v34 = 1;
      v33 = 0;
      LODWORD(v36) = 1;
      v35 = 0;
      v38 = 1;
      v37 = 0;
      v26 = 0;
      v21 = sub_BD2910((__int64)a3);
      sub_D15BF0(a2, v5, *a3, v21, (__int64)&v27, &v29, (__int64)&v31, (__int64)&v35, (__int64)&v26);
      v22 = v30;
      v23 = v38 <= 0x40;
      v30 = 0;
      *(_DWORD *)(a1 + 8) = v22;
      *(_QWORD *)a1 = v29;
      if ( !v23 && v37 )
        j_j___libc_free_0_0(v37);
      if ( (unsigned int)v36 > 0x40 && v35 )
        j_j___libc_free_0_0(v35);
      if ( v34 > 0x40 && v33 )
        j_j___libc_free_0_0(v33);
      if ( v32 > 0x40 && v31 )
        j_j___libc_free_0_0(v31);
      if ( v30 > 0x40 && v29 )
        j_j___libc_free_0_0(v29);
      if ( v28 > 0x40 && v27 )
        j_j___libc_free_0_0(v27);
    }
  }
  else
  {
    *(_DWORD *)(a1 + 8) = v11;
    if ( (unsigned int)v11 > 0x40 )
    {
      sub_C43690(a1, -1, 1);
    }
    else
    {
      v13 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
      if ( !(_DWORD)v11 )
        v13 = 0;
      *(_QWORD *)a1 = v13;
    }
  }
  return a1;
}
