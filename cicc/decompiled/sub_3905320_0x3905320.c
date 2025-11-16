// Function: sub_3905320
// Address: 0x3905320
//
__int64 __fastcall sub_3905320(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdi
  _BYTE **v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // r12d
  unsigned int v13; // eax
  __int64 *v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdi
  unsigned __int64 *v19; // r13
  __int64 v20; // rbx
  unsigned __int64 *v21; // rbx
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdi
  void (*v26)(); // rax
  _QWORD v27[2]; // [rsp+0h] [rbp-130h] BYREF
  _QWORD v28[2]; // [rsp+10h] [rbp-120h] BYREF
  __int16 v29; // [rsp+20h] [rbp-110h]
  const char *v30; // [rsp+30h] [rbp-100h] BYREF
  char *v31; // [rsp+38h] [rbp-F8h]
  __int16 v32; // [rsp+40h] [rbp-F0h]
  const char **v33; // [rsp+50h] [rbp-E0h] BYREF
  char *v34; // [rsp+58h] [rbp-D8h]
  _QWORD v35[2]; // [rsp+60h] [rbp-D0h] BYREF
  unsigned __int64 *v36; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+78h] [rbp-B8h]
  _BYTE v38[176]; // [rsp+80h] [rbp-B0h] BYREF

  v27[0] = a2;
  v27[1] = a3;
  v36 = (unsigned __int64 *)v38;
  v37 = 0x400000000LL;
  while ( 1 )
  {
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
    {
      v23 = *(_QWORD *)(a1 + 8);
      v32 = 1283;
      v30 = "expected string in '";
      v31 = (char *)v27;
      v33 = &v30;
      LOWORD(v35[0]) = 770;
      v34 = "' directive";
      v12 = sub_3909CF0(v23, &v33, 0, 0, v4, v5);
      goto LABEL_15;
    }
    v6 = *(_QWORD *)(a1 + 8);
    v33 = (const char **)v35;
    v7 = (_BYTE **)&v33;
    v34 = 0;
    LOBYTE(v35[0]) = 0;
    v12 = (*(__int64 (__fastcall **)(__int64, const char ***))(*(_QWORD *)v6 + 160LL))(v6, &v33);
    if ( (_BYTE)v12 )
      goto LABEL_13;
    v13 = v37;
    if ( (unsigned int)v37 >= HIDWORD(v37) )
    {
      v7 = 0;
      sub_12BE710((__int64)&v36, 0, v8, v9, v10, v11);
      v13 = v37;
    }
    v14 = (__int64 *)&v36[4 * v13];
    if ( v14 )
    {
      *v14 = (__int64)(v14 + 2);
      v7 = (_BYTE **)v33;
      sub_3901E80(v14, v33, (__int64)&v34[(_QWORD)v33]);
      v13 = v37;
    }
    v15 = *(_QWORD *)(a1 + 8);
    LODWORD(v37) = v13 + 1;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v15 + 40LL))(v15) + 8) == 9 )
      break;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
    {
      v18 = *(_QWORD *)(a1 + 8);
      v28[0] = "unexpected token in '";
      v28[1] = v27;
      v29 = 1283;
      v30 = (const char *)v28;
      v32 = 770;
      v31 = "' directive";
      v12 = sub_3909CF0(v18, &v30, 0, 0, v16, v17);
LABEL_13:
      if ( v33 != v35 )
        j_j___libc_free_0((unsigned __int64)v33);
LABEL_15:
      v19 = v36;
      v20 = (unsigned int)v37;
      goto LABEL_16;
    }
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    if ( v33 != v35 )
      j_j___libc_free_0((unsigned __int64)v33);
  }
  if ( v33 != v35 )
  {
    v7 = (_BYTE **)(v35[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v33);
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, _BYTE **))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8), v7);
  v19 = v36;
  v20 = (unsigned int)v37;
  v25 = v24;
  v26 = *(void (**)())(*(_QWORD *)v24 + 200LL);
  if ( v26 != nullsub_582 )
  {
    ((void (__fastcall *)(__int64, unsigned __int64 *, _QWORD))v26)(v25, v36, (unsigned int)v37);
    v19 = v36;
    v20 = (unsigned int)v37;
  }
LABEL_16:
  v21 = &v19[4 * v20];
  if ( v21 != v19 )
  {
    do
    {
      v21 -= 4;
      if ( (unsigned __int64 *)*v21 != v21 + 2 )
        j_j___libc_free_0(*v21);
    }
    while ( v21 != v19 );
    v19 = v36;
  }
  if ( v19 != (unsigned __int64 *)v38 )
    _libc_free((unsigned __int64)v19);
  return v12;
}
