// Function: sub_221C600
// Address: 0x221c600
//
_QWORD *__fastcall sub_221C600(
        __int64 a1,
        _QWORD *a2,
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
  char v19; // al
  int v21; // eax
  int v22; // eax
  _QWORD *v23; // [rsp+8h] [rbp-68h]
  int v24; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v25[15]; // [rsp+34h] [rbp-3Ch] BYREF

  v25[0] = 0;
  v11 = sub_221C220(a1, a2, a3, a4, a5, &v24, 0, 9999, 4u, a6, v25);
  v13 = v12;
  v14 = v11;
  v15 = v12 | a3 & 0xFFFFFFFF00000000LL;
  v16 = v25[0];
  if ( v25[0] )
  {
    *a7 |= 4u;
  }
  else
  {
    v16 = (unsigned int)(v24 - 1900);
    a2 = (_QWORD *)a8;
    if ( v24 < 0 )
      v16 = (unsigned int)(v24 + 100);
    *(_DWORD *)(a8 + 20) = v16;
  }
  v17 = v13 == -1;
  v18 = v17 & (v11 != 0);
  if ( v18 )
  {
    v17 = 0;
    if ( v11[2] >= v11[3] )
    {
      v22 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD *, unsigned __int64, __int64, _QWORD *))(*v11 + 72LL))(
              v11,
              a2,
              v15,
              v16,
              v11);
      v14 = 0;
      if ( v22 == -1 )
        v17 = v18;
      else
        v14 = v11;
    }
  }
  v19 = a5 == -1;
  if ( a4 )
  {
    if ( a5 == -1 )
    {
      v19 = 0;
      if ( a4[2] >= a4[3] )
      {
        v23 = v14;
        v21 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD *))(*a4 + 72LL))(a4, a2);
        v14 = v23;
        v19 = v21 == -1;
      }
    }
  }
  if ( v17 == v19 )
    *a7 |= 2u;
  return v14;
}
