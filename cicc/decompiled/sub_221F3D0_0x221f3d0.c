// Function: sub_221F3D0
// Address: 0x221f3d0
//
char *__fastcall sub_221F3D0(
        __int64 a1,
        char *a2,
        unsigned __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        _DWORD *a7,
        _DWORD *a8)
{
  __int64 v12; // rax
  char *v13; // r12
  int v14; // edx
  unsigned __int8 v15; // bl
  char v16; // al
  unsigned __int8 v18; // r15

  v12 = sub_22311C0(a6 + 208);
  v13 = sub_221D830(a1, a2, a3, (char *)a4, a5, a6, a7, a8, *(char **)(*(_QWORD *)(v12 + 16) + 32LL));
  v15 = v14 == -1;
  if ( (v15 & (v13 != 0)) != 0 )
  {
    v18 = v15 & (v13 != 0);
    v15 = 0;
    if ( *((_QWORD *)v13 + 2) >= *((_QWORD *)v13 + 3)
      && (*(unsigned int (__fastcall **)(char *))(*(_QWORD *)v13 + 72LL))(v13) == -1 )
    {
      v15 = v18;
      v13 = 0;
    }
  }
  v16 = a5 == -1;
  if ( a4 )
  {
    if ( a5 == -1 )
    {
      v16 = 0;
      if ( *(_QWORD *)(a4 + 16) >= *(_QWORD *)(a4 + 24) )
        v16 = (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)a4 + 72LL))(a4) == -1;
    }
  }
  if ( v15 == v16 )
    *a7 |= 2u;
  return v13;
}
