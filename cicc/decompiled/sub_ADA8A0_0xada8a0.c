// Function: sub_ADA8A0
// Address: 0xada8a0
//
__int64 __fastcall sub_ADA8A0(unsigned __int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rdx
  int v5; // eax
  unsigned int v7; // r15d
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rax
  int v12; // ecx
  unsigned __int8 *v13; // r15
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 *v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rbx
  int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // r12
  unsigned int v22; // r15d
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // r8
  unsigned __int64 v26; // rax
  __int64 *v27; // [rsp+0h] [rbp-C0h]
  __int64 v28; // [rsp+8h] [rbp-B8h]
  __int64 v29; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v30; // [rsp+18h] [rbp-A8h]
  __int64 v31; // [rsp+20h] [rbp-A0h]
  unsigned int v32; // [rsp+28h] [rbp-98h]
  char v33; // [rsp+30h] [rbp-90h]
  __int64 *v34; // [rsp+40h] [rbp-80h] BYREF
  __int64 v35; // [rsp+48h] [rbp-78h]
  _BYTE v36[112]; // [rsp+50h] [rbp-70h] BYREF

  v4 = a2;
  if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(a2 + 16);
  if ( *(_BYTE *)a1 != 5 )
    return sub_AD4B40(0x32u, a1, (__int64 **)a2, a3);
  v5 = *(unsigned __int16 *)(a1 + 2);
  if ( v5 == 34 )
  {
    v7 = *(_DWORD *)(v4 + 8) >> 8;
    v8 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    v9 = sub_BD5C60(a1, a2, v4);
    v10 = sub_BCE3C0(v9, v7);
    if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
      v10 = sub_BCDA70(v10, *(unsigned int *)(a2 + 32));
    v11 = sub_ADA8A0(v8, v10, 0);
    v34 = (__int64 *)v36;
    v12 = 0;
    v13 = (unsigned __int8 *)v11;
    v14 = 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    v35 = 0x800000000LL;
    v15 = (__int64 *)v36;
    v16 = (__int64 *)(a1 + v14);
    v17 = -v14;
    v18 = v17 >> 5;
    if ( (unsigned __int64)v17 > 0x100 )
    {
      sub_C8D5F0(&v34, v36, v17 >> 5, 8);
      v12 = v35;
      v15 = &v34[(unsigned int)v35];
    }
    if ( (__int64 *)a1 != v16 )
    {
      do
      {
        if ( v15 )
          *v15 = *v16;
        v16 += 4;
        ++v15;
      }
      while ( (__int64 *)a1 != v16 );
      v12 = v35;
    }
    LODWORD(v35) = v12 + v18;
    sub_BB52D0(&v29, a1);
    v27 = v34;
    v19 = *(_BYTE *)(a1 + 1) >> 1;
    v28 = (unsigned int)v35;
    v20 = sub_BB5290(a1, a1, v34);
    v21 = sub_AD9FD0(v20, v13, v27, v28, (v19 << 31 >> 31) & 3, (__int64)&v29, 0);
    if ( v33 )
    {
      v33 = 0;
      if ( v32 > 0x40 && v31 )
        j_j___libc_free_0_0(v31);
      if ( v30 > 0x40 && v29 )
        j_j___libc_free_0_0(v29);
    }
    if ( v34 != (__int64 *)v36 )
      _libc_free(v34, v13);
    return v21;
  }
  else
  {
    if ( v5 != 49 )
      return sub_AD4B40(0x32u, a1, (__int64 **)a2, a3);
    v22 = *(_DWORD *)(v4 + 8) >> 8;
    v23 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    v24 = sub_BD5C60(a1, a2, v4);
    v25 = sub_BCE3C0(v24, v22);
    if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
      v25 = sub_BCDA70(v25, *(unsigned int *)(a2 + 32));
    v26 = sub_ADA8A0(v23, v25, 0);
    return sub_AD4B40(0x31u, v26, (__int64 **)a2, a3);
  }
}
