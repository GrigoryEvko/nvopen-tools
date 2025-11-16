// Function: sub_370C3D0
// Address: 0x370c3d0
//
__int64 __fastcall sub_370C3D0(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rdx
  _DWORD *v7; // r13
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned int *v14; // r13
  unsigned __int64 v15; // rax
  __int64 *v16; // rsi
  unsigned __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // r12
  unsigned __int64 v21; // [rsp+0h] [rbp-860h]
  _BYTE *v22; // [rsp+10h] [rbp-850h]
  __int64 *v23; // [rsp+28h] [rbp-838h]
  _DWORD *v24; // [rsp+30h] [rbp-830h]
  unsigned __int64 v25; // [rsp+38h] [rbp-828h]
  _QWORD v26[2]; // [rsp+40h] [rbp-820h] BYREF
  _QWORD v27[2]; // [rsp+50h] [rbp-810h] BYREF
  unsigned int v28; // [rsp+64h] [rbp-7FCh] BYREF
  __int64 v29; // [rsp+68h] [rbp-7F8h] BYREF
  _BYTE *v30; // [rsp+70h] [rbp-7F0h] BYREF
  __int64 v31; // [rsp+78h] [rbp-7E8h]
  _BYTE v32[48]; // [rsp+80h] [rbp-7E0h] BYREF
  _BYTE v33[1968]; // [rsp+B0h] [rbp-7B0h] BYREF

  v27[0] = a3;
  v30 = v32;
  v31 = 0x400000000LL;
  v27[1] = a4;
  v26[0] = a5;
  v26[1] = a6;
  sub_37FBE60(a1, a2, &v30);
  sub_CC1970((__int64)v33);
  sub_CC1970((__int64)v33);
  v6 = 4;
  if ( a2 <= 4 )
    v6 = a2;
  sub_CC19D0((__int64)v33, a1, v6);
  v7 = v30;
  v21 = a2 - 4;
  v23 = (__int64 *)((char *)a1 + 4);
  v22 = &v30[12 * (unsigned int)v31];
  if ( v30 == v22 )
    goto LABEL_19;
  v8 = 0;
  while ( 1 )
  {
    sub_CC19D0((__int64)v33, (__int64 *)((char *)v23 + v8), (unsigned int)(v7[1] - v8));
    v9 = v27;
    if ( *v7 == 1 )
      v9 = v26;
    v10 = *v9;
    v11 = (unsigned int)v7[2];
    v25 = v9[1];
    v12 = (unsigned int)v7[1];
    v13 = (__int64)v23 + 4 * v11 + v12;
    if ( (__int64 *)v13 != (__int64 *)((char *)v23 + v12) )
      break;
LABEL_17:
    v8 = (unsigned int)(v12 + 4 * v11);
    v7 += 3;
    if ( v22 == (_BYTE *)v7 )
    {
      v23 = (__int64 *)((char *)v23 + v8);
      v21 -= v8;
LABEL_19:
      sub_CC19D0((__int64)v33, v23, v21);
      sub_CC21A0((__int64)v33, &v29, 8u);
      v19 = v29;
      goto LABEL_21;
    }
  }
  v24 = v7;
  v14 = (unsigned int *)((char *)v23 + v12);
  while ( 1 )
  {
    v18 = *v14;
    v28 = v18;
    if ( v18 <= 0xFFF )
    {
      v17 = 4;
      v16 = (__int64 *)&v28;
      goto LABEL_13;
    }
    LODWORD(v29) = 0;
    v15 = (v18 & 0x7FFFFFFF) - 4096;
    if ( v15 >= v25 )
      break;
    v16 = (__int64 *)(v10 + 8 * v15);
    if ( !*v16 )
      break;
    v17 = 8;
LABEL_13:
    ++v14;
    sub_CC19D0((__int64)v33, v16, v17);
    if ( (unsigned int *)v13 == v14 )
    {
      v7 = v24;
      LODWORD(v12) = v24[1];
      LODWORD(v11) = v24[2];
      goto LABEL_17;
    }
  }
  v19 = 0;
LABEL_21:
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return v19;
}
