// Function: sub_2237090
// Address: 0x2237090
//
_QWORD *__fastcall sub_2237090(
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
  char v14; // al
  unsigned __int64 v15; // rdi
  bool v17; // zf
  char v18; // al
  _QWORD *v19; // rax
  int v20; // edx
  char v21; // [rsp+7h] [rbp-69h]
  _QWORD *v22; // [rsp+10h] [rbp-60h]
  int v23; // [rsp+18h] [rbp-58h]
  char v24; // [rsp+18h] [rbp-58h]
  __int64 v25[8]; // [rsp+30h] [rbp-40h] BYREF

  v25[0] = (__int64)&unk_4FD67D8;
  sub_2215AB0(v25, 0x20u);
  v12 = sub_2236190(a1, a2, a3, a4, a5, a6, a7, v25);
  v23 = v13;
  v22 = v12;
  v25[1] = sub_2208E60(a1, a2);
  sub_22540D0(v25[0], a8, a7);
  v24 = v23 == -1;
  v21 = v24 & (v12 != 0);
  if ( v21 )
  {
    v24 = 0;
    if ( v12[2] >= v12[3] )
    {
      v17 = (*(unsigned int (__fastcall **)(_QWORD *))(*v12 + 72LL))(v12) == -1;
      v18 = v21;
      if ( !v17 )
        v18 = 0;
      v24 = v18;
      v19 = 0;
      if ( !v17 )
        v19 = v12;
      v22 = v19;
    }
  }
  v14 = (_DWORD)a5 == -1;
  if ( a4 )
  {
    if ( (_DWORD)a5 == -1 )
    {
      v14 = 0;
      if ( a4[2] >= a4[3] )
        v14 = (*(unsigned int (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4) == -1;
    }
  }
  if ( v14 == v24 )
    *a7 |= 2u;
  v15 = v25[0] - 24;
  if ( (_UNKNOWN *)(v25[0] - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v20 = _InterlockedExchangeAdd((volatile signed __int32 *)(v25[0] - 8), 0xFFFFFFFF);
    }
    else
    {
      v20 = *(_DWORD *)(v25[0] - 8);
      *(_DWORD *)(v25[0] - 8) = v20 - 1;
    }
    if ( v20 <= 0 )
      j_j___libc_free_0_1(v15);
  }
  return v22;
}
