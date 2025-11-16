// Function: sub_2243450
// Address: 0x2243450
//
_QWORD *__fastcall sub_2243450(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        int a5,
        __int64 a6,
        _DWORD *a7,
        __int64 a8)
{
  _QWORD *v11; // rbp
  unsigned int v12; // edx
  unsigned int v13; // eax
  _QWORD *v14; // r8
  unsigned __int64 v15; // rdx
  __int64 v16; // rcx
  char v17; // bl
  char v18; // r15
  __int64 v19; // rcx
  int *v21; // rax
  int v22; // eax
  unsigned int *v23; // rax
  _QWORD *v24; // [rsp+8h] [rbp-68h]
  unsigned __int64 v25; // [rsp+8h] [rbp-68h]
  int v26; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v27[15]; // [rsp+34h] [rbp-3Ch] BYREF

  v27[0] = 0;
  v11 = sub_2243170(a1, a2, a3, a4, a5, &v26, 0, 9999, 4u, a6, v27);
  v13 = v12;
  v14 = v11;
  v15 = v12 | a3 & 0xFFFFFFFF00000000LL;
  v16 = v27[0];
  if ( v27[0] )
  {
    *a7 |= 4u;
  }
  else
  {
    v16 = (unsigned int)(v26 - 1900);
    a2 = a8;
    if ( v26 < 0 )
      v16 = (unsigned int)(v26 + 100);
    *(_DWORD *)(a8 + 20) = v16;
  }
  v17 = v13 == -1;
  v18 = v17 & (v11 != 0);
  if ( v18 )
  {
    v23 = (unsigned int *)v11[2];
    if ( (unsigned __int64)v23 >= v11[3] )
    {
      v25 = v15;
      v13 = (*(__int64 (__fastcall **)(_QWORD *, __int64, unsigned __int64, __int64, _QWORD *))(*v11 + 72LL))(
              v11,
              a2,
              v15,
              v16,
              v11);
      v15 = v25;
    }
    else
    {
      v13 = *v23;
    }
    v17 = 0;
    v14 = 0;
    if ( v13 == -1 )
      v17 = v18;
    else
      v14 = v11;
  }
  LOBYTE(v13) = a5 == -1;
  v19 = v13;
  if ( a4 && a5 == -1 )
  {
    v21 = (int *)a4[2];
    if ( (unsigned __int64)v21 >= a4[3] )
    {
      v24 = v14;
      v22 = (*(__int64 (__fastcall **)(_QWORD *, __int64, unsigned __int64, __int64))(*a4 + 72LL))(a4, a2, v15, v19);
      v14 = v24;
    }
    else
    {
      v22 = *v21;
    }
    LOBYTE(v19) = v22 == -1;
  }
  if ( (_BYTE)v19 == v17 )
    *a7 |= 2u;
  return v14;
}
