// Function: sub_224ACC0
// Address: 0x224acc0
//
_QWORD *__fastcall sub_224ACC0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        __int64 a8)
{
  _QWORD *v12; // rbx
  int v13; // edx
  _QWORD *v14; // r15
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  unsigned __int64 v18; // rdi
  int *v20; // rax
  int v21; // eax
  int *v22; // rax
  int v23; // eax
  bool v24; // zf
  int v25; // edx
  int v26; // [rsp+18h] [rbp-58h]
  unsigned __int8 v27; // [rsp+18h] [rbp-58h]
  __int64 v28[8]; // [rsp+30h] [rbp-40h] BYREF

  v28[0] = (__int64)&unk_4FD67D8;
  sub_2215AB0(v28, 0x20u);
  v12 = sub_2249BE0(a1, a2, a3, a4, a5, a6, a7, v28);
  v26 = v13;
  v14 = v12;
  v28[1] = sub_2208E60(a1, a2);
  v15 = sub_22540D0(v28[0], a8, a7);
  v27 = v26 == -1;
  v16 = v27;
  if ( (v27 & (v12 != 0)) != 0 )
  {
    v22 = (int *)v12[2];
    if ( (unsigned __int64)v22 >= v12[3] )
      v23 = (*(__int64 (__fastcall **)(_QWORD *))(*v12 + 72LL))(v12);
    else
      v23 = *v22;
    v24 = v23 == -1;
    LODWORD(v15) = v27 & (v12 != 0);
    v14 = 0;
    if ( !v24 )
    {
      v14 = v12;
      LODWORD(v15) = 0;
    }
    v27 = v15;
  }
  LOBYTE(v15) = (_DWORD)a5 == -1;
  v17 = (unsigned int)v15;
  if ( a4 && (_DWORD)a5 == -1 )
  {
    v20 = (int *)a4[2];
    if ( (unsigned __int64)v20 >= a4[3] )
      v21 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*a4 + 72LL))(a4, v16, v17);
    else
      v21 = *v20;
    LOBYTE(v17) = v21 == -1;
  }
  if ( v27 == (_BYTE)v17 )
    *a7 |= 2u;
  v18 = v28[0] - 24;
  if ( (_UNKNOWN *)(v28[0] - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v25 = _InterlockedExchangeAdd((volatile signed __int32 *)(v28[0] - 8), 0xFFFFFFFF);
    }
    else
    {
      v25 = *(_DWORD *)(v28[0] - 8);
      *(_DWORD *)(v28[0] - 8) = v25 - 1;
    }
    if ( v25 <= 0 )
      j_j___libc_free_0_1(v18);
  }
  return v14;
}
