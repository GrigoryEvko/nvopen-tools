// Function: sub_222A840
// Address: 0x222a840
//
_QWORD *__fastcall sub_222A840(
        __int64 a1,
        _QWORD *a2,
        unsigned __int64 a3,
        _QWORD *a4,
        _QWORD *a5,
        __int64 a6,
        _DWORD *a7,
        _DWORD *a8,
        char a9,
        char a10)
{
  __int64 v14; // rax
  _QWORD *v15; // r14
  unsigned int v16; // edx
  unsigned int v17; // eax
  __int64 v18; // rcx
  _QWORD *v19; // r8
  unsigned __int64 v20; // rdx
  char v21; // bl
  char v22; // r15
  __int64 v23; // rcx
  int *v25; // rax
  int v26; // eax
  unsigned int *v27; // rax
  _QWORD *v28; // [rsp+8h] [rbp-70h]
  unsigned __int64 v29; // [rsp+8h] [rbp-70h]
  wchar_t s; // [rsp+30h] [rbp-48h] BYREF
  int v31; // [rsp+34h] [rbp-44h]
  int v32; // [rsp+38h] [rbp-40h]
  int v33; // [rsp+3Ch] [rbp-3Ch]

  v14 = sub_2243120(a6 + 208);
  *a7 = 0;
  s = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 80LL))(v14, 37);
  if ( a10 )
  {
    v32 = a9;
    v31 = a10;
    v33 = 0;
  }
  else
  {
    v31 = a9;
    v32 = 0;
  }
  v15 = sub_22290E0(a1, a2, a3, a4, a5, a6, a7, a8, &s);
  v17 = v16;
  v18 = v16;
  v19 = v15;
  v20 = v16 | a3 & 0xFFFFFFFF00000000LL;
  v21 = v17 == -1;
  v22 = v21 & (v15 != 0);
  if ( v22 )
  {
    v27 = (unsigned int *)v15[2];
    if ( (unsigned __int64)v27 >= v15[3] )
    {
      v29 = v20;
      v17 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD *, unsigned __int64, __int64, _QWORD *))(*v15 + 72LL))(
              v15,
              a2,
              v20,
              v18,
              v15);
      v20 = v29;
    }
    else
    {
      v17 = *v27;
    }
    v21 = 0;
    v19 = 0;
    if ( v17 == -1 )
      v21 = v22;
    else
      v19 = v15;
  }
  LOBYTE(v17) = (_DWORD)a5 == -1;
  v23 = v17;
  if ( a4 && (_DWORD)a5 == -1 )
  {
    v25 = (int *)a4[2];
    if ( (unsigned __int64)v25 >= a4[3] )
    {
      v28 = v19;
      v26 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD *, unsigned __int64, __int64))(*a4 + 72LL))(a4, a2, v20, v23);
      v19 = v28;
    }
    else
    {
      v26 = *v25;
    }
    LOBYTE(v23) = v26 == -1;
  }
  if ( (_BYTE)v23 == v21 )
    *a7 |= 2u;
  return v19;
}
