// Function: sub_ABEA30
// Address: 0xabea30
//
__int64 __fastcall sub_ABEA30(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // dl
  _QWORD *v4; // rcx
  unsigned int v5; // ebx
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rdi
  bool v11; // cc
  unsigned __int8 v12; // dl
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int v16; // edx
  __int64 v18; // [rsp+8h] [rbp-B8h]
  __int64 v19; // [rsp+10h] [rbp-B0h]
  __int64 v20; // [rsp+18h] [rbp-A8h]
  __int64 v22; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-88h]
  __int64 v24; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v25; // [rsp+48h] [rbp-78h]
  __int64 v26; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v27; // [rsp+58h] [rbp-68h]
  __int64 v28; // [rsp+60h] [rbp-60h]
  unsigned int v29; // [rsp+68h] [rbp-58h]
  __int64 v30; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+78h] [rbp-48h]
  __int64 v32; // [rsp+80h] [rbp-40h]
  int v33; // [rsp+88h] [rbp-38h]

  v3 = *(_BYTE *)(a2 - 16);
  v19 = a2 - 16;
  if ( (v3 & 2) != 0 )
  {
    v4 = *(_QWORD **)(a2 - 32);
    v5 = *(_DWORD *)(a2 - 24) >> 1;
  }
  else
  {
    v4 = (_QWORD *)(v19 - 8LL * ((v3 >> 2) & 0xF));
    v5 = ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) >> 1;
  }
  v6 = *(_QWORD *)(*v4 + 136LL);
  v7 = *(_QWORD *)(v4[1] + 136LL);
  v31 = *(_DWORD *)(v7 + 32);
  if ( v31 > 0x40 )
    sub_C43780(&v30, v7 + 24);
  else
    v30 = *(_QWORD *)(v7 + 24);
  v27 = *(_DWORD *)(v6 + 32);
  if ( v27 > 0x40 )
    sub_C43780(&v26, v6 + 24);
  else
    v26 = *(_QWORD *)(v6 + 24);
  sub_AADC30(a1, (__int64)&v26, &v30);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v5 > 1 )
  {
    v8 = v5 - 2;
    v9 = 24;
    v20 = 16 * v8 + 40;
    do
    {
      v12 = *(_BYTE *)(a2 - 16);
      if ( (v12 & 2) != 0 )
        v13 = *(_QWORD *)(a2 - 32);
      else
        v13 = v19 - 8LL * ((v12 >> 2) & 0xF);
      v14 = *(_QWORD *)(*(_QWORD *)(v13 + v9 - 8) + 136LL);
      v15 = *(_QWORD *)(*(_QWORD *)(v13 + v9) + 136LL);
      v25 = *(_DWORD *)(v15 + 32);
      if ( v25 > 0x40 )
      {
        v18 = v14;
        sub_C43780(&v24, v15 + 24);
        v14 = v18;
      }
      else
      {
        v24 = *(_QWORD *)(v15 + 24);
      }
      v23 = *(_DWORD *)(v14 + 32);
      if ( v23 > 0x40 )
        sub_C43780(&v22, v14 + 24);
      else
        v22 = *(_QWORD *)(v14 + 24);
      sub_AADC30((__int64)&v26, (__int64)&v22, &v24);
      sub_AB3510((__int64)&v30, a1, (__int64)&v26, 0);
      if ( *(_DWORD *)(a1 + 8) > 0x40u && *(_QWORD *)a1 )
        j_j___libc_free_0_0(*(_QWORD *)a1);
      v11 = *(_DWORD *)(a1 + 24) <= 0x40u;
      *(_QWORD *)a1 = v30;
      v16 = v31;
      v31 = 0;
      *(_DWORD *)(a1 + 8) = v16;
      if ( v11 || (v10 = *(_QWORD *)(a1 + 16)) == 0 )
      {
        *(_QWORD *)(a1 + 16) = v32;
        *(_DWORD *)(a1 + 24) = v33;
      }
      else
      {
        j_j___libc_free_0_0(v10);
        v11 = v31 <= 0x40;
        *(_QWORD *)(a1 + 16) = v32;
        *(_DWORD *)(a1 + 24) = v33;
        if ( !v11 && v30 )
          j_j___libc_free_0_0(v30);
      }
      if ( v29 > 0x40 && v28 )
        j_j___libc_free_0_0(v28);
      if ( v27 > 0x40 && v26 )
        j_j___libc_free_0_0(v26);
      if ( v23 > 0x40 && v22 )
        j_j___libc_free_0_0(v22);
      if ( v25 > 0x40 && v24 )
        j_j___libc_free_0_0(v24);
      v9 += 16;
    }
    while ( v9 != v20 );
  }
  return a1;
}
