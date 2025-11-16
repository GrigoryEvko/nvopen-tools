// Function: sub_EDDB80
// Address: 0xeddb80
//
__int64 *__fastcall sub_EDDB80(__int64 *a1, unsigned __int64 *a2, __int64 a3, _QWORD *a4)
{
  __int64 v7; // r12
  __int64 v8; // rcx
  bool v9; // cc
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rsi
  _QWORD *v22; // rax
  _QWORD *v23; // r12
  _QWORD *v24; // rdi
  _QWORD *v25; // rdi
  _QWORD *v26; // rdi
  __int64 v27; // r13
  __int64 v28; // r12
  _QWORD *v29; // rax
  unsigned __int64 v30; // rdi
  __int64 v31; // r13
  _QWORD *v32; // rax
  unsigned __int64 v33; // rdi
  __int64 v34; // [rsp+10h] [rbp-350h]
  __int64 v35; // [rsp+18h] [rbp-348h]
  __int64 v36; // [rsp+20h] [rbp-340h]
  __int64 v37; // [rsp+20h] [rbp-340h]
  __int64 v38; // [rsp+20h] [rbp-340h]
  __int64 v39; // [rsp+28h] [rbp-338h]
  __int64 v40; // [rsp+28h] [rbp-338h]
  _QWORD *v42; // [rsp+38h] [rbp-328h] BYREF
  _QWORD v43[2]; // [rsp+40h] [rbp-320h] BYREF
  char v44; // [rsp+50h] [rbp-310h] BYREF
  char v45; // [rsp+130h] [rbp-230h]
  unsigned __int64 v46; // [rsp+140h] [rbp-220h] BYREF
  _QWORD v47[2]; // [rsp+148h] [rbp-218h] BYREF
  _BYTE v48[224]; // [rsp+158h] [rbp-208h] BYREF
  _BYTE *v49; // [rsp+238h] [rbp-128h]
  __int64 v50; // [rsp+240h] [rbp-120h]
  _BYTE v51[168]; // [rsp+248h] [rbp-118h] BYREF
  _BYTE *v52; // [rsp+2F0h] [rbp-70h]
  __int64 v53; // [rsp+2F8h] [rbp-68h]
  _BYTE v54[96]; // [rsp+300h] [rbp-60h] BYREF

  v42 = a4;
  v7 = *a4;
  v42 = a4 + 1;
  v8 = a4[1];
  v42 = a4 + 2;
  v9 = *a2 <= 1;
  v36 = v8;
  v39 = a4[2];
  v42 = a4 + 3;
  if ( v9 )
  {
    v34 = 0;
    v35 = 0;
  }
  else
  {
    v10 = (__int64)(a4 + 4);
    v11 = a4[3];
    v12 = (__int64)(a4 + 5);
    v42 = (_QWORD *)v10;
    v35 = v11;
    v34 = *(_QWORD *)(v12 - 8);
    v42 = (_QWORD *)v12;
  }
  sub_C16C80((__int64)v43, (__int64)&v42);
  v15 = v45 & 1;
  v45 = (2 * v15) | v45 & 0xFD;
  if ( (_BYTE)v15 )
  {
    *a1 = v43[0] | 1LL;
  }
  else
  {
    sub_ED6600((__int64)(a2 + 1), (__int64)v43, v15, (unsigned int)(2 * v15), v13, v14);
    v46 = *a2;
    v47[0] = v48;
    v47[1] = 0x1C00000000LL;
    if ( *((_DWORD *)a2 + 4) )
      sub_ED6600((__int64)v47, (__int64)(a2 + 1), v17, v18, v19, v20);
    v21 = (__int64)v42;
    v49 = v51;
    v50 = 0x100000000LL;
    v52 = v54;
    v53 = 0x600000000LL;
    v22 = sub_ED8DB0((__int64 *)(a3 + v7), (__int64)v42, a3, (__int64)&v46);
    v23 = (_QWORD *)a2[31];
    a2[31] = (unsigned __int64)v22;
    if ( v23 )
    {
      v24 = (_QWORD *)v23[58];
      if ( v24 != v23 + 60 )
        _libc_free(v24, v21);
      v25 = (_QWORD *)v23[35];
      if ( v25 != v23 + 37 )
        _libc_free(v25, v21);
      v26 = (_QWORD *)v23[5];
      if ( v26 != v23 + 7 )
        _libc_free(v26, v21);
      v21 = 536;
      j_j___libc_free_0(v23, 536);
    }
    if ( v52 != v54 )
      _libc_free(v52, v21);
    if ( v49 != v51 )
      _libc_free(v49, v21);
    if ( (_BYTE *)v47[0] != v48 )
      _libc_free(v47[0], v21);
    v27 = *(_QWORD *)(a3 + v39 + 8);
    v28 = a3 + v39 + 16;
    v37 = a3 + v36;
    v40 = *(_QWORD *)(a3 + v39);
    v29 = (_QWORD *)sub_22077B0(48);
    if ( v29 )
    {
      v21 = v37;
      v29[1] = v27;
      v29[2] = v28;
      *v29 = v40;
      v29[3] = a3;
      v29[5] = v37;
    }
    v30 = a2[32];
    a2[32] = (unsigned __int64)v29;
    if ( v30 )
    {
      v21 = 48;
      j_j___libc_free_0(v30, 48);
    }
    if ( *a2 > 1 )
    {
      v31 = *(_QWORD *)(a3 + v34 + 8);
      v38 = *(_QWORD *)(a3 + v34);
      v32 = (_QWORD *)sub_22077B0(48);
      if ( v32 )
      {
        v21 = a3 + v35;
        v32[1] = v31;
        v32[2] = a3 + v34 + 16;
        *v32 = v38;
        v32[3] = a3;
        v32[5] = a3 + v35;
      }
      v33 = a2[33];
      a2[33] = (unsigned __int64)v32;
      if ( v33 )
      {
        v21 = 48;
        j_j___libc_free_0(v33, 48);
      }
    }
    *a1 = 1;
    if ( (v45 & 2) != 0 )
      sub_EDDB10(v43, v21);
    if ( (v45 & 1) != 0 )
    {
      if ( v43[0] )
        (*(void (**)(void))(*(_QWORD *)v43[0] + 8LL))();
    }
    else if ( (char *)v43[0] != &v44 )
    {
      _libc_free(v43[0], v21);
    }
  }
  return a1;
}
