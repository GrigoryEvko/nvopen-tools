// Function: sub_32086D0
// Address: 0x32086d0
//
__int64 __fastcall sub_32086D0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // r13
  void (*v9)(); // rax
  __int64 v10; // rdi
  void (*v11)(); // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // rdi
  void (*v15)(); // rax
  volatile signed __int32 *v16; // r12
  __int64 result; // rax
  __int64 v18; // [rsp+10h] [rbp-120h]
  __int64 v20; // [rsp+28h] [rbp-108h] BYREF
  _BYTE v21[10]; // [rsp+36h] [rbp-FAh] BYREF
  _QWORD v22[4]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v23; // [rsp+60h] [rbp-D0h]
  _QWORD v24[2]; // [rsp+70h] [rbp-C0h] BYREF
  volatile signed __int32 *v25; // [rsp+80h] [rbp-B0h]
  __int64 v26; // [rsp+A8h] [rbp-88h]
  unsigned __int64 v27[2]; // [rsp+B0h] [rbp-80h] BYREF
  _BYTE v28[16]; // [rsp+C0h] [rbp-70h] BYREF
  char v29; // [rsp+D0h] [rbp-60h]
  char v30; // [rsp+D1h] [rbp-5Fh]
  __int64 v31; // [rsp+D8h] [rbp-58h]
  _QWORD *v32; // [rsp+E0h] [rbp-50h]
  __int64 v33; // [rsp+E8h] [rbp-48h]
  __int64 v34; // [rsp+F0h] [rbp-40h]

  v7 = sub_31F8790(a1, 4359, a3, a4, a5);
  v8 = *(_QWORD *)(a1 + 528);
  v18 = v7;
  v9 = *(void (**)())(*(_QWORD *)v8 + 120LL);
  v30 = 1;
  v27[0] = (unsigned __int64)"Type";
  v29 = 3;
  if ( v9 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v9)(v8, v27, 1);
    v8 = *(_QWORD *)(a1 + 528);
  }
  LODWORD(v27[0]) = sub_3206530(a1, a2, 0);
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v8 + 536LL))(v8, LODWORD(v27[0]), 4);
  v10 = *(_QWORD *)(a1 + 528);
  v11 = *(void (**)())(*(_QWORD *)v10 + 120LL);
  v30 = 1;
  v27[0] = (unsigned __int64)"Value";
  v29 = 3;
  if ( v11 != nullsub_98 )
    ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v11)(v10, v27, 1);
  sub_3719220(v24, v21, 10, 1);
  v32 = v24;
  v27[0] = (unsigned __int64)v28;
  v27[1] = 0x200000000LL;
  v31 = 0;
  v33 = 0;
  v34 = 0;
  v23 = 257;
  sub_3702790(&v20, v27, a3, v22);
  if ( (v20 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  (*(void (__fastcall **)(_QWORD, _BYTE *, __int64))(**(_QWORD **)(a1 + 528) + 520LL))(*(_QWORD *)(a1 + 528), v21, v26);
  v14 = *(__int64 **)(a1 + 528);
  v15 = *(void (**)())(*v14 + 120);
  v22[0] = "Name";
  v23 = 259;
  if ( v15 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v15)(v14, v22, 1);
    v14 = *(__int64 **)(a1 + 528);
  }
  sub_31F4F00(v14, *(const void **)a4, *(_QWORD *)(a4 + 8), 3840, v12, v13);
  sub_31F8930(a1, v18);
  if ( (_BYTE *)v27[0] != v28 )
    _libc_free(v27[0]);
  v16 = v25;
  result = (__int64)&unk_4A352E0;
  v24[0] = &unk_4A352E0;
  if ( v25 )
  {
    if ( &_pthread_key_create )
    {
      result = (unsigned int)_InterlockedExchangeAdd(v25 + 2, 0xFFFFFFFF);
    }
    else
    {
      result = *((unsigned int *)v25 + 2);
      *((_DWORD *)v25 + 2) = result - 1;
    }
    if ( (_DWORD)result == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 16LL))(v16);
      if ( &_pthread_key_create )
      {
        result = (unsigned int)_InterlockedExchangeAdd(v16 + 3, 0xFFFFFFFF);
      }
      else
      {
        result = *((unsigned int *)v16 + 3);
        *((_DWORD *)v16 + 3) = result - 1;
      }
      if ( (_DWORD)result == 1 )
        return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 24LL))(v16);
    }
  }
  return result;
}
