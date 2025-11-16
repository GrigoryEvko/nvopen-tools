// Function: sub_34CF150
// Address: 0x34cf150
//
__int64 __fastcall sub_34CF150(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 (__fastcall *v4)(unsigned __int64 *); // r12
  __int64 v5; // r12
  unsigned __int64 v6; // r14
  __int64 (__fastcall *v7)(__int64); // rax
  unsigned __int64 *v8; // r13
  unsigned __int64 *v9; // r12
  __int64 (*v10)(void); // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // r13
  _QWORD *v18; // r12
  __int64 (__fastcall *v19)(_QWORD *); // rax
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 (__fastcall *v26)(__int64, unsigned __int64 *, __int64); // r12
  __int64 v27; // r12
  _QWORD *v28; // rax
  void (__fastcall *v29)(__int64, char); // rdx
  void (__fastcall *v30)(__int64, char); // rdx
  void (__fastcall *v31)(__int64, char); // rax
  bool v32; // si
  __int64 result; // rax
  __int64 v34; // rdi
  __int64 v35; // [rsp+0h] [rbp-B0h]
  __int64 (__fastcall *v36)(unsigned __int64 *, __int64, __int64, __int64, __int64); // [rsp+8h] [rbp-A8h]
  void *v37; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v38; // [rsp+18h] [rbp-98h]
  __int16 v39; // [rsp+30h] [rbp-80h]
  unsigned __int64 v40[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v41[12]; // [rsp+50h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a1 + 520);
  v4 = *(__int64 (__fastcall **)(unsigned __int64 *))(*(_QWORD *)(a1 + 8) + 80LL);
  if ( v4 )
  {
    a2 = (__int64)&v37;
    v37 = *(void **)(a1 + 512);
    v39 = 261;
    v38 = v3;
    sub_CC9F70((__int64)v40, &v37);
    v5 = v4(v40);
    if ( (_QWORD *)v40[0] != v41 )
    {
      a2 = v41[0] + 1LL;
      j_j___libc_free_0(v40[0]);
    }
  }
  else
  {
    v5 = 0;
  }
  v6 = *(_QWORD *)(a1 + 664);
  *(_QWORD *)(a1 + 664) = v5;
  if ( v6 )
  {
    v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL);
    if ( v7 == sub_C11FA0 )
    {
      v8 = *(unsigned __int64 **)(v6 + 232);
      v9 = *(unsigned __int64 **)(v6 + 224);
      *(_QWORD *)v6 = &unk_49E3560;
      if ( v8 != v9 )
      {
        do
        {
          if ( *v9 )
            j_j___libc_free_0(*v9);
          v9 += 3;
        }
        while ( v8 != v9 );
        v9 = *(unsigned __int64 **)(v6 + 224);
      }
      if ( v9 )
        j_j___libc_free_0((unsigned __int64)v9);
      sub_C7D6A0(*(_QWORD *)(v6 + 200), 8LL * *(unsigned int *)(v6 + 216), 4);
      sub_C7D6A0(*(_QWORD *)(v6 + 168), 8LL * *(unsigned int *)(v6 + 184), 4);
      a2 = 248;
      j_j___libc_free_0(v6);
    }
    else
    {
      v7(v6);
    }
  }
  v10 = *(__int64 (**)(void))(*(_QWORD *)(a1 + 8) + 64LL);
  if ( v10 )
    v11 = v10();
  else
    v11 = 0;
  v12 = *(_QWORD *)(a1 + 672);
  *(_QWORD *)(a1 + 672) = v11;
  if ( v12 )
  {
    a2 = 48;
    j_j___libc_free_0(v12);
  }
  v13 = *(_QWORD *)(a1 + 600);
  v14 = *(_QWORD *)(a1 + 608);
  v35 = *(_QWORD *)(a1 + 576);
  v15 = *(_QWORD *)(a1 + 520);
  v36 = *(__int64 (__fastcall **)(unsigned __int64 *, __int64, __int64, __int64, __int64))(*(_QWORD *)(a1 + 8) + 88LL);
  v16 = *(_QWORD *)(a1 + 568);
  if ( v36 )
  {
    v37 = *(void **)(a1 + 512);
    v39 = 261;
    v38 = v15;
    sub_CC9F70((__int64)v40, &v37);
    a2 = v16;
    v17 = v36(v40, v16, v35, v13, v14);
    if ( (_QWORD *)v40[0] != v41 )
    {
      a2 = v41[0] + 1LL;
      j_j___libc_free_0(v40[0]);
    }
  }
  else
  {
    v17 = 0;
  }
  v18 = *(_QWORD **)(a1 + 680);
  *(_QWORD *)(a1 + 680) = v17;
  if ( v18 )
  {
    v19 = *(__int64 (__fastcall **)(_QWORD *))(*v18 + 8LL);
    if ( v19 == sub_C12070 )
    {
      v20 = v18[34];
      *v18 = &unk_49E41D0;
      if ( (_QWORD *)v20 != v18 + 36 )
        j_j___libc_free_0(v20);
      v21 = v18[12];
      if ( (_QWORD *)v21 != v18 + 14 )
        j_j___libc_free_0(v21);
      v22 = v18[8];
      if ( (_QWORD *)v22 != v18 + 10 )
        j_j___libc_free_0(v22);
      v23 = v18[1];
      if ( (_QWORD *)v23 != v18 + 3 )
        j_j___libc_free_0(v23);
      j_j___libc_free_0((unsigned __int64)v18);
    }
    else
    {
      ((void (__fastcall *)(_QWORD *, __int64))v19)(v18, a2);
    }
  }
  v24 = *(_QWORD *)(a1 + 520);
  v25 = *(_QWORD *)(a1 + 664);
  v26 = *(__int64 (__fastcall **)(__int64, unsigned __int64 *, __int64))(*(_QWORD *)(a1 + 8) + 48LL);
  if ( v26 )
  {
    v37 = *(void **)(a1 + 512);
    v39 = 261;
    v38 = v24;
    sub_CC9F70((__int64)v40, &v37);
    v27 = v26(v25, v40, a1 + 976);
    if ( (_QWORD *)v40[0] != v41 )
      j_j___libc_free_0(v40[0]);
  }
  else
  {
    v27 = 0;
  }
  if ( *(int *)(a1 + 856) > 0 )
    *(_QWORD *)(v27 + 384) = *(_QWORD *)(a1 + 856);
  v28 = *(_QWORD **)v27;
  if ( (*(_BYTE *)(a1 + 876) & 2) != 0 )
  {
    v29 = (void (__fastcall *)(__int64, char))v28[9];
    if ( v29 == sub_106E240 )
    {
      *(_BYTE *)(v27 + 392) = 0;
      v30 = (void (__fastcall *)(__int64, char))v28[10];
      if ( v30 == sub_106E250 )
      {
LABEL_40:
        *(_BYTE *)(v27 + 393) = 0;
        goto LABEL_41;
      }
    }
    else
    {
      v29(v27, 0);
      v28 = *(_QWORD **)v27;
      v30 = *(void (__fastcall **)(__int64, char))(*(_QWORD *)v27 + 80LL);
      if ( v30 == sub_106E250 )
        goto LABEL_40;
    }
    v30(v27, 0);
    v28 = *(_QWORD **)v27;
  }
LABEL_41:
  v31 = (void (__fastcall *)(__int64, char))v28[11];
  v32 = (*(_BYTE *)(a1 + 977) & 0x10) != 0;
  if ( v31 == sub_106E260 )
    *(_BYTE *)(v27 + 394) = v32;
  else
    v31(v27, v32);
  *(_BYTE *)(v27 + 187) = (*(_BYTE *)(a1 + 1224) & 2) != 0;
  result = *(unsigned int *)(a1 + 972);
  if ( (_DWORD)result )
    *(_DWORD *)(v27 + 336) = result;
  v34 = *(_QWORD *)(a1 + 656);
  *(_QWORD *)(a1 + 656) = v27;
  if ( v34 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v34 + 8LL))(v34);
  return result;
}
