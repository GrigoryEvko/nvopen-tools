// Function: sub_2EBFBC0
// Address: 0x2ebfbc0
//
_WORD *__fastcall sub_2EBFBC0(_QWORD *a1)
{
  _WORD *v1; // r15
  __int64 v3; // rax
  _WORD *v4; // rax
  _WORD *v5; // r14
  int v6; // ebx
  __int64 (*v7)(); // rax

  if ( *((_BYTE *)a1 + 176) )
    return (_WORD *)a1[23];
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 16LL) + 200LL))(*(_QWORD *)(*a1 + 16LL));
  v4 = (_WORD *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v3 + 72LL))(v3, *a1);
  v1 = v4;
  if ( *v4 )
  {
    v5 = v4;
    v6 = 0;
    do
    {
      v7 = *(__int64 (**)())(**(_QWORD **)(*a1 + 16LL) + 496LL);
      if ( v7 != sub_2EBDF70 && (unsigned __int8)v7() )
        sub_2EBF8D0(*(_QWORD **)(*a1 + 32LL), (unsigned __int16)*v5);
      v5 = &v1[++v6];
    }
    while ( *v5 );
  }
  return v1;
}
