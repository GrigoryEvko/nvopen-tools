// Function: sub_17E6910
// Address: 0x17e6910
//
void __fastcall sub_17E6910(__int64 a1, unsigned int a2)
{
  __int64 v3; // rsi
  _QWORD *v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 *v7; // rax
  unsigned int v8; // r15d
  __int64 v9; // rsi
  unsigned int v10; // r8d
  __int64 v11; // r14
  char *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // [rsp+0h] [rbp-100h]
  __int64 *v17; // [rsp+8h] [rbp-F8h]
  __int64 v18; // [rsp+10h] [rbp-F0h]
  _QWORD v19[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int16 v20; // [rsp+40h] [rbp-C0h]
  _QWORD v21[2]; // [rsp+50h] [rbp-B0h] BYREF
  __int16 v22; // [rsp+60h] [rbp-A0h]
  _QWORD v23[2]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v24; // [rsp+80h] [rbp-80h]
  __int64 v25[2]; // [rsp+90h] [rbp-70h] BYREF
  _QWORD v26[2]; // [rsp+A0h] [rbp-60h] BYREF
  _QWORD v27[10]; // [rsp+B0h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a1 + 40) + 24LL * a2;
  v4 = *(_QWORD **)(a1 + 368);
  v5 = 0;
  if ( v4 )
  {
    if ( a2 )
      v5 = -1431655765 * (unsigned int)((__int64)(v4[4] - v4[3]) >> 3);
    else
      v5 = -1431655765 * (unsigned int)((__int64)(v4[1] - *v4) >> 3);
  }
  v6 = *(__int64 **)v3;
  v17 = *(__int64 **)(v3 + 8);
  if ( ((__int64)v17 - *(_QWORD *)v3) >> 3 == v5 )
  {
    if ( v6 != v17 )
    {
      v7 = &qword_4FA5780;
      if ( a2 == 1 )
        v7 = &qword_4FA56A0;
      v8 = 0;
      v16 = v7;
      do
      {
        v9 = *v6;
        v10 = v8++;
        ++v6;
        sub_1695280(*(__int64 ***)(a1 + 8), v9, a1 + 344, a2, v10, *((_DWORD *)v16 + 40));
      }
      while ( v17 != v6 );
    }
  }
  else
  {
    v11 = **(_QWORD **)(a1 + 8);
    v12 = (char *)sub_1649960(*(_QWORD *)a1);
    if ( v12 )
    {
      v25[0] = (__int64)v26;
      sub_17E2210(v25, v12, (__int64)&v12[v13]);
    }
    else
    {
      v25[1] = 0;
      v25[0] = (__int64)v26;
      LOBYTE(v26[0]) = 0;
    }
    LODWORD(v18) = a2;
    v19[0] = "Inconsistent number of value sites for kind = ";
    v22 = 770;
    v19[1] = v18;
    v20 = 2307;
    v21[0] = v19;
    v21[1] = " in ";
    v23[0] = v21;
    v14 = *(_QWORD *)(a1 + 8);
    v24 = 1026;
    v15 = *(_QWORD *)(v14 + 176);
    v23[1] = v25;
    v27[1] = 0x100000012LL;
    v27[2] = v15;
    v27[0] = &unk_49ECF40;
    v27[3] = v23;
    sub_16027F0(v11, (__int64)v27);
    if ( (_QWORD *)v25[0] != v26 )
      j_j___libc_free_0(v25[0], v26[0] + 1LL);
  }
}
