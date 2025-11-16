// Function: sub_2FEEAE0
// Address: 0x2feeae0
//
_QWORD *__fastcall sub_2FEEAE0(__int64 a1, char a2)
{
  unsigned int v2; // eax
  __int64 (*v3)(void); // rax
  _QWORD *(__fastcall *v5)(__int64, char); // rax
  _QWORD *(*v6)(); // [rsp+8h] [rbp-18h] BYREF

  v6 = sub_2FEDDE0;
  *(_QWORD *)(__readfsqword(0) - 24) = &v6;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_2FEDE30;
  if ( !&_pthread_key_create )
  {
    v2 = -1;
LABEL_11:
    sub_4264C5(v2);
  }
  v2 = pthread_once(&dword_5026ED0, init_routine);
  if ( v2 )
    goto LABEL_11;
  v3 = (__int64 (*)(void))qword_5023860[1];
  if ( v3 != sub_2FEDDD0 )
    return (_QWORD *)v3();
  v5 = *(_QWORD *(__fastcall **)(__int64, char))(*(_QWORD *)a1 + 312LL);
  if ( v5 != sub_2FEE3D0 )
    return v5(a1, a2);
  if ( a2 )
    return (_QWORD *)sub_2F504C0();
  return sub_2F42900();
}
