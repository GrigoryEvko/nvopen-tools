// Function: sub_BC2570
// Address: 0xbc2570
//
__int64 __fastcall sub_BC2570(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  _QWORD *v6; // rcx
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD *v10; // rbx
  __int64 v11; // r12
  _QWORD *v12; // rax
  _QWORD *v13; // rdi
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rsi
  char *v17; // rsi
  unsigned int (__fastcall *v18)(_QWORD, _QWORD); // rax
  void **v19; // rax
  __int64 v20; // rcx
  void **v21; // rdx
  void **v23; // rsi
  char v24; // [rsp+17h] [rbp-109h]
  _QWORD *v26; // [rsp+20h] [rbp-100h]
  _QWORD *v29; // [rsp+40h] [rbp-E0h]
  _QWORD *v30; // [rsp+48h] [rbp-D8h]
  __int64 v31; // [rsp+50h] [rbp-D0h] BYREF
  _QWORD *v32; // [rsp+58h] [rbp-C8h] BYREF
  _QWORD v33[4]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+80h] [rbp-A0h]
  char v35[8]; // [rsp+90h] [rbp-90h] BYREF
  __int64 v36; // [rsp+98h] [rbp-88h]
  char v37; // [rsp+ACh] [rbp-74h]
  __int64 v38; // [rsp+C8h] [rbp-58h]
  char v39; // [rsp+DCh] [rbp-44h]

  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_QWORD *)a1 = 1;
  v31 = *(_QWORD *)(sub_BC1CD0(a4, &unk_4F8A320, a3) + 8);
  v24 = *(_BYTE *)(a3 + 128);
  sub_B2B9F0(a3, qword_4F80F48[8]);
  sub_C85EE0(v33);
  v6 = (_QWORD *)*a2;
  v33[3] = a3;
  v34 = 0;
  v30 = v6;
  v33[0] = &unk_49DB080;
  v33[2] = &v31;
  v26 = (_QWORD *)a2[1];
  if ( v6 == v26 )
    goto LABEL_22;
  while ( 1 )
  {
    v34 = *v30;
    if ( (unsigned __int8)sub_BBBC50(&v31, v34, a3) )
      break;
LABEL_21:
    if ( v26 == ++v30 )
      goto LABEL_22;
  }
  (*(void (__fastcall **)(char *, _QWORD, __int64, __int64))(*(_QWORD *)*v30 + 16LL))(v35, *v30, a3, a4);
  sub_BBE020(a4, a3, (__int64)v35, v7);
  if ( v31 )
  {
    v10 = *(_QWORD **)(v31 + 432);
    v29 = &v10[4 * *(unsigned int *)(v31 + 440)];
    if ( v10 != v29 )
    {
      v11 = *v30;
      do
      {
        v32 = 0;
        v12 = (_QWORD *)sub_22077B0(16);
        if ( v12 )
        {
          v12[1] = a3;
          *v12 = &unk_49DB0A8;
        }
        v13 = v32;
        v32 = v12;
        if ( v13 )
          (*(void (__fastcall **)(_QWORD *))(*v13 + 8LL))(v13);
        v14 = v10;
        v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 32LL))(v11);
        if ( (v10[3] & 2) == 0 )
          v14 = (_QWORD *)*v10;
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD **, char *))(v10[3] & 0xFFFFFFFFFFFFFFF8LL))(
          v14,
          v16,
          v15,
          &v32,
          v35);
        if ( v32 )
          (*(void (__fastcall **)(_QWORD *))(*v32 + 8LL))(v32);
        v10 += 4;
      }
      while ( v29 != v10 );
    }
  }
  v17 = v35;
  sub_BBADB0(a1, (__int64)v35, v8, v9);
  v18 = (unsigned int (__fastcall *)(_QWORD, _QWORD))a2[3];
  if ( !v18 || (v17 = 0, !v18(a2[4], 0)) )
  {
    if ( !v39 )
      _libc_free(v38, v17);
    if ( !v37 )
      _libc_free(v36, v17);
    goto LABEL_21;
  }
  if ( !v39 )
    _libc_free(v38, 0);
  if ( v37 )
  {
LABEL_22:
    if ( *(_DWORD *)(a1 + 68) == *(_DWORD *)(a1 + 72) )
      goto LABEL_33;
    goto LABEL_23;
  }
  _libc_free(v36, 0);
  if ( *(_DWORD *)(a1 + 68) == *(_DWORD *)(a1 + 72) )
  {
LABEL_33:
    if ( *(_BYTE *)(a1 + 28) )
    {
      v19 = *(void ***)(a1 + 8);
      v23 = &v19[*(unsigned int *)(a1 + 20)];
      LODWORD(v20) = *(_DWORD *)(a1 + 20);
      v21 = v19;
      if ( v19 == v23 )
        goto LABEL_39;
      while ( *v21 != &unk_4F82400 )
      {
        if ( v23 == ++v21 )
        {
LABEL_27:
          while ( *v19 != &unk_4F82420 )
          {
            if ( v21 == ++v19 )
              goto LABEL_39;
          }
          goto LABEL_28;
        }
      }
      goto LABEL_28;
    }
    if ( sub_C8CA60(a1, &unk_4F82400, v5, v6) )
      goto LABEL_28;
  }
LABEL_23:
  if ( !*(_BYTE *)(a1 + 28) )
    goto LABEL_41;
  v19 = *(void ***)(a1 + 8);
  v20 = *(unsigned int *)(a1 + 20);
  v21 = &v19[v20];
  if ( v21 != v19 )
    goto LABEL_27;
LABEL_39:
  if ( *(_DWORD *)(a1 + 16) > (unsigned int)v20 )
  {
    *(_DWORD *)(a1 + 20) = v20 + 1;
    *v21 = &unk_4F82420;
    ++*(_QWORD *)a1;
  }
  else
  {
LABEL_41:
    sub_C8CC70(a1, &unk_4F82420);
  }
LABEL_28:
  v33[0] = &unk_49DB080;
  nullsub_162(v33);
  sub_B2B9F0(a3, v24);
  return a1;
}
