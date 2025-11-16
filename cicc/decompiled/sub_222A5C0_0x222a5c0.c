// Function: sub_222A5C0
// Address: 0x222a5c0
//
_QWORD *__fastcall sub_222A5C0(
        __int64 a1,
        _QWORD *a2,
        unsigned __int64 a3,
        _QWORD *a4,
        _QWORD *a5,
        __int64 a6,
        _DWORD *a7,
        _DWORD *a8)
{
  __int64 v12; // rax
  _QWORD *v13; // r14
  unsigned int v14; // edx
  unsigned int v15; // eax
  __int64 v16; // rcx
  _QWORD *v17; // r8
  unsigned __int64 v18; // rdx
  unsigned __int8 v19; // bl
  __int64 v20; // rcx
  int *v22; // rax
  int v23; // eax
  unsigned __int8 v24; // r15
  unsigned int *v25; // rax
  _QWORD *v26; // [rsp+0h] [rbp-58h]
  unsigned __int64 v27; // [rsp+0h] [rbp-58h]

  v12 = sub_2244AF0(a6 + 208);
  v13 = sub_22290E0(a1, a2, a3, a4, a5, a6, a7, a8, *(wchar_t **)(*(_QWORD *)(v12 + 16) + 32LL));
  v15 = v14;
  v16 = v14;
  v17 = v13;
  v18 = v14 | a3 & 0xFFFFFFFF00000000LL;
  v19 = v15 == -1;
  if ( (v19 & (v13 != 0)) != 0 )
  {
    v24 = v19 & (v13 != 0);
    v25 = (unsigned int *)v13[2];
    if ( (unsigned __int64)v25 >= v13[3] )
    {
      v27 = v18;
      v15 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD *, unsigned __int64, __int64, _QWORD *))(*v13 + 72LL))(
              v13,
              a2,
              v18,
              v16,
              v13);
      v18 = v27;
    }
    else
    {
      v15 = *v25;
    }
    v19 = 0;
    v17 = 0;
    if ( v15 == -1 )
      v19 = v24;
    else
      v17 = v13;
  }
  LOBYTE(v15) = (_DWORD)a5 == -1;
  v20 = v15;
  if ( a4 && (_DWORD)a5 == -1 )
  {
    v22 = (int *)a4[2];
    if ( (unsigned __int64)v22 >= a4[3] )
    {
      v26 = v17;
      v23 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD *, unsigned __int64, __int64))(*a4 + 72LL))(a4, a2, v18, v20);
      v17 = v26;
    }
    else
    {
      v23 = *v22;
    }
    LOBYTE(v20) = v23 == -1;
  }
  if ( v19 == (_BYTE)v20 )
    *a7 |= 2u;
  return v17;
}
