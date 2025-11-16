// Function: sub_39D2910
// Address: 0x39d2910
//
void __fastcall sub_39D2910(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rcx
  char v4; // al
  __int64 v5; // rcx
  char v6; // al
  __int64 v7; // rcx
  char v8; // al
  __int64 v9; // rcx
  char v10; // al
  _BOOL8 v11; // rcx
  char v12; // al
  _BOOL8 v13; // rcx
  char v14; // al
  _BOOL8 v15; // rcx
  char v16; // al
  __int64 v17; // rcx
  char v18; // al
  __int64 v19; // rcx
  char v20; // al
  _BOOL8 v21; // rcx
  char v22; // al
  __int64 v23; // rcx
  char v24; // al
  __int64 v25; // rcx
  char v26; // al
  __int64 v27; // rcx
  char v28; // al
  _BOOL8 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rax
  const char *v32; // rax
  __int64 v33; // rdx
  void (__fastcall *v34)(__int64, unsigned __int64 **); // rax
  __int64 v35; // rax
  char v36; // [rsp+17h] [rbp-99h] BYREF
  __int64 v37; // [rsp+18h] [rbp-98h] BYREF
  __int128 v38; // [rsp+20h] [rbp-90h] BYREF
  const char *v39; // [rsp+30h] [rbp-80h] BYREF
  __int64 v40; // [rsp+38h] [rbp-78h]
  _BYTE v41[16]; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 *v42; // [rsp+50h] [rbp-60h] BYREF
  __int64 v43; // [rsp+58h] [rbp-58h]
  __int64 v44; // [rsp+60h] [rbp-50h] BYREF
  __int64 v45; // [rsp+68h] [rbp-48h]
  __int128 v46; // [rsp+70h] [rbp-40h]

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = *(_BYTE *)a2 ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "isFrameAddressTaken",
         0,
         v3,
         &v39,
         &v42) )
  {
    sub_39D0760(a1, (_BYTE *)a2);
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_BYTE *)a2 = 0;
  }
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *(_BYTE *)(a2 + 1) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "isReturnAddressTaken",
         0,
         v5,
         &v39,
         &v42) )
  {
    sub_39D0760(a1, (_BYTE *)(a2 + 1));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_BYTE *)(a2 + 1) = 0;
  }
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = *(_BYTE *)(a2 + 2) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "hasStackMap",
         0,
         v7,
         &v39,
         &v42) )
  {
    sub_39D0760(a1, (_BYTE *)(a2 + 2));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_BYTE *)(a2 + 2) = 0;
  }
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 )
    v9 = *(_BYTE *)(a2 + 3) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "hasPatchPoint",
         0,
         v9,
         &v39,
         &v42) )
  {
    sub_39D0760(a1, (_BYTE *)(a2 + 3));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_BYTE *)(a2 + 3) = 0;
  }
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v11 = 0;
  if ( v10 )
    v11 = *(_QWORD *)(a2 + 8) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "stackSize",
         0,
         v11,
         &v39,
         &v42) )
  {
    sub_39D05E0(a1, (_QWORD *)(a2 + 8));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_QWORD *)(a2 + 8) = 0;
  }
  v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v13 = 0;
  if ( v12 )
    v13 = *(_DWORD *)(a2 + 16) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "offsetAdjustment",
         0,
         v13,
         &v36,
         &v37) )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    {
      v41[0] = 0;
      v39 = v41;
      v40 = 0;
      LODWORD(v46) = 1;
      v45 = 0;
      v44 = 0;
      v43 = 0;
      v42 = (unsigned __int64 *)&unk_49EFBE0;
      *((_QWORD *)&v46 + 1) = &v39;
      v35 = sub_16E4080(a1);
      sub_16E5AA0((int *)(a2 + 16), v35, (__int64)&v42);
      if ( v45 != v43 )
        sub_16E7BA0((__int64 *)&v42);
      v38 = **((_OWORD **)&v46 + 1);
      (*(void (__fastcall **)(__int64, __int128 *, _QWORD))(*(_QWORD *)a1 + 216LL))(a1, &v38, 0);
      sub_16E7BC0((__int64 *)&v42);
      if ( v39 != v41 )
        j_j___libc_free_0((unsigned __int64)v39);
    }
    else
    {
      v30 = *(_QWORD *)a1;
      v38 = 0u;
      (*(void (__fastcall **)(__int64, __int128 *, _QWORD))(v30 + 216))(a1, &v38, 0);
      v31 = sub_16E4080(a1);
      v32 = sub_16E5AB0(v38, v31, (_DWORD *)(a2 + 16));
      v40 = v33;
      v39 = v32;
      if ( v33 )
      {
        v34 = *(void (__fastcall **)(__int64, unsigned __int64 **))(*(_QWORD *)a1 + 232LL);
        LOWORD(v44) = 261;
        v42 = (unsigned __int64 *)&v39;
        v34(a1, &v42);
      }
    }
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v37);
  }
  else if ( v36 )
  {
    *(_DWORD *)(a2 + 16) = 0;
  }
  v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v15 = 0;
  if ( v14 )
    v15 = *(_DWORD *)(a2 + 20) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "maxAlignment",
         0,
         v15,
         &v39,
         &v42) )
  {
    sub_39D02E0(a1, (unsigned int *)(a2 + 20));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_DWORD *)(a2 + 20) = 0;
  }
  v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v17 = 0;
  if ( v16 )
    v17 = *(_BYTE *)(a2 + 24) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "adjustsStack",
         0,
         v17,
         &v39,
         &v42) )
  {
    sub_39D0760(a1, (_BYTE *)(a2 + 24));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_BYTE *)(a2 + 24) = 0;
  }
  v18 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v19 = 0;
  if ( v18 )
    v19 = *(_BYTE *)(a2 + 25) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "hasCalls",
         0,
         v19,
         &v39,
         &v42) )
  {
    sub_39D0760(a1, (_BYTE *)(a2 + 25));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_BYTE *)(a2 + 25) = 0;
  }
  v42 = (unsigned __int64 *)&v44;
  v43 = 0;
  LOBYTE(v44) = 0;
  v46 = 0;
  sub_39D1940(a1, (__int64)"stackProtector", a2 + 32, (__int64)&v42, 0);
  if ( v42 != (unsigned __int64 *)&v44 )
    j_j___libc_free_0((unsigned __int64)v42);
  v20 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v21 = 0;
  if ( v20 )
    v21 = *(_DWORD *)(a2 + 80) == -1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "maxCallFrameSize",
         0,
         v21,
         &v39,
         &v42) )
  {
    sub_39D02E0(a1, (unsigned int *)(a2 + 80));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_DWORD *)(a2 + 80) = -1;
  }
  v22 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v23 = 0;
  if ( v22 )
    v23 = *(_BYTE *)(a2 + 84) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "hasOpaqueSPAdjustment",
         0,
         v23,
         &v39,
         &v42) )
  {
    sub_39D0760(a1, (_BYTE *)(a2 + 84));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_BYTE *)(a2 + 84) = 0;
  }
  v24 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v25 = 0;
  if ( v24 )
    v25 = *(_BYTE *)(a2 + 85) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "hasVAStart",
         0,
         v25,
         &v39,
         &v42) )
  {
    sub_39D0760(a1, (_BYTE *)(a2 + 85));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_BYTE *)(a2 + 85) = 0;
  }
  v26 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v27 = 0;
  if ( v26 )
    v27 = *(_BYTE *)(a2 + 86) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "hasMustTailInVarArgFunc",
         0,
         v27,
         &v39,
         &v42) )
  {
    sub_39D0760(a1, (_BYTE *)(a2 + 86));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_BYTE *)(a2 + 86) = 0;
  }
  v28 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v29 = 0;
  if ( v28 )
    v29 = *(_DWORD *)(a2 + 88) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, const char **, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "localFrameSize",
         0,
         v29,
         &v39,
         &v42) )
  {
    sub_39D02E0(a1, (unsigned int *)(a2 + 88));
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  else if ( (_BYTE)v39 )
  {
    *(_DWORD *)(a2 + 88) = 0;
  }
  v42 = (unsigned __int64 *)&v44;
  v43 = 0;
  LOBYTE(v44) = 0;
  v46 = 0;
  sub_39D1940(a1, (__int64)"savePoint", a2 + 96, (__int64)&v42, 0);
  if ( v42 != (unsigned __int64 *)&v44 )
    j_j___libc_free_0((unsigned __int64)v42);
  v42 = (unsigned __int64 *)&v44;
  v43 = 0;
  LOBYTE(v44) = 0;
  v46 = 0;
  sub_39D1940(a1, (__int64)"restorePoint", a2 + 144, (__int64)&v42, 0);
  if ( v42 != (unsigned __int64 *)&v44 )
    j_j___libc_free_0((unsigned __int64)v42);
}
