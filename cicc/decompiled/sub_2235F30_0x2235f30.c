// Function: sub_2235F30
// Address: 0x2235f30
//
_QWORD *__fastcall sub_2235F30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        _DWORD *a8)
{
  __int64 v12; // rax
  _QWORD *v13; // r12
  int v14; // edx
  unsigned __int8 v15; // bl
  char v16; // al
  unsigned __int8 v18; // r15

  v12 = sub_22311C0((_QWORD *)(a6 + 208), a2);
  v13 = (_QWORD *)sub_2234C80(a1, a2, a3, (__int64)a4, a5, a6, a7, a8, *(char **)(*(_QWORD *)(v12 + 16) + 32LL));
  v15 = v14 == -1;
  if ( (v15 & (v13 != 0)) != 0 )
  {
    v18 = v15 & (v13 != 0);
    v15 = 0;
    if ( v13[2] >= v13[3] && (*(unsigned int (__fastcall **)(_QWORD *))(*v13 + 72LL))(v13) == -1 )
    {
      v15 = v18;
      v13 = 0;
    }
  }
  v16 = (_DWORD)a5 == -1;
  if ( a4 )
  {
    if ( (_DWORD)a5 == -1 )
    {
      v16 = 0;
      if ( a4[2] >= a4[3] )
        v16 = (*(unsigned int (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4) == -1;
    }
  }
  if ( v15 == v16 )
    *a7 |= 2u;
  return v13;
}
