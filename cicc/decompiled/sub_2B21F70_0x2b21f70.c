// Function: sub_2B21F70
// Address: 0x2b21f70
//
__int64 __fastcall sub_2B21F70(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE **v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // r13
  int v12; // ecx
  __int64 *v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r10
  _QWORD *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // eax
  int v23; // ecx
  unsigned __int8 **v24; // rbx
  __int64 v25; // rax
  __int64 *v26; // r12
  __int64 v27; // rcx
  __int128 v28; // [rsp-18h] [rbp-128h]
  __int64 v29; // [rsp+8h] [rbp-108h]
  unsigned __int8 *v30; // [rsp+18h] [rbp-F8h]
  __int64 v31; // [rsp+28h] [rbp-E8h]
  _QWORD v32[2]; // [rsp+30h] [rbp-E0h] BYREF
  _QWORD v33[3]; // [rsp+40h] [rbp-D0h] BYREF
  char *v34; // [rsp+58h] [rbp-B8h]
  char v35; // [rsp+68h] [rbp-A8h] BYREF
  char *v36; // [rsp+88h] [rbp-88h]
  char v37; // [rsp+98h] [rbp-78h] BYREF

  if ( a3 )
  {
    v33[0] = a3;
    v6 = (_BYTE **)v33;
    v7 = 1;
  }
  else
  {
    v18 = *(_QWORD **)a1;
    v6 = **(_BYTE ****)a1;
    v7 = v18[1];
  }
  LOBYTE(v8) = sub_99AF90(v6, v7);
  v9 = v8;
  if ( !(_DWORD)v8 )
    return 0;
  v12 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v12 - 17) > 1 )
  {
    if ( (_BYTE)v12 != 14 )
    {
      v14 = (__int64 *)a2;
      goto LABEL_8;
    }
    v15 = a2;
    v19 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 3344LL);
LABEL_19:
    v20 = sub_9208B0(v19, v15);
    v33[1] = v21;
    v33[0] = v20;
    v22 = sub_CA1930(v33);
    v14 = (__int64 *)sub_BCCE00(*(_QWORD **)a2, v22);
    v23 = *(unsigned __int8 *)(a2 + 8);
    if ( (unsigned int)(v23 - 17) <= 1 )
    {
      BYTE4(v31) = (_BYTE)v23 == 18;
      LODWORD(v31) = *(_DWORD *)(a2 + 32);
      v14 = (__int64 *)sub_BCE1B0(v14, v31);
    }
    goto LABEL_8;
  }
  v13 = *(__int64 **)(a2 + 16);
  v14 = (__int64 *)a2;
  v15 = *v13;
  if ( *(_BYTE *)(*v13 + 8) == 14 )
  {
    v19 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 3344LL);
    if ( (unsigned __int8)(v12 - 17) >= 2u )
      v15 = a2;
    goto LABEL_19;
  }
LABEL_8:
  v32[0] = v14;
  v32[1] = v14;
  *((_QWORD *)&v28 + 1) = 1;
  *(_QWORD *)&v28 = 0;
  sub_DF8CB0((__int64)v33, v9, (__int64)v14, (char *)v32, 2, 0, 0, v28);
  v16 = sub_DFD690(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 3296LL), (__int64)v33);
  v17 = v16;
  if ( a3 && BYTE4(v9) )
  {
    v24 = (*(_BYTE *)(a3 + 7) & 0x40) != 0
        ? *(unsigned __int8 ***)(a3 - 8)
        : (unsigned __int8 **)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
    v25 = *(_QWORD *)(a1 + 8);
    v29 = v17;
    v30 = *v24;
    v26 = *(__int64 **)(v25 + 3296);
    sub_BCB2A0(*(_QWORD **)(v25 + 3440));
    v27 = sub_DFD2D0(v26, (unsigned int)*v30 - 29, a2);
    v16 = v29 - v27;
    if ( __OFSUB__(v29, v27) )
    {
      v16 = 0x8000000000000000LL;
      if ( v27 <= 0 )
        v16 = 0x7FFFFFFFFFFFFFFFLL;
    }
  }
  v10 = v16;
  if ( v36 != &v37 )
    _libc_free((unsigned __int64)v36);
  if ( v34 != &v35 )
    _libc_free((unsigned __int64)v34);
  return v10;
}
