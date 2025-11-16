// Function: sub_1D446C0
// Address: 0x1d446c0
//
void __fastcall sub_1D446C0(__int64 a1, __int64 *a2)
{
  _QWORD *i; // rbx
  void (*v3)(); // rax
  __int64 *v4; // rax
  __int64 *v5; // r14
  _QWORD *j; // rbx

  if ( (unsigned __int8)sub_1D12E90((__int64)a2) || (v4 = sub_16BDF00((__int64 *)(a1 + 320), a2), v5 = v4, a2 == v4) )
  {
    for ( i = *(_QWORD **)(a1 + 664); i; i = (_QWORD *)i[1] )
    {
      while ( 1 )
      {
        v3 = *(void (**)())(*i + 24LL);
        if ( v3 != nullsub_685 )
          break;
        i = (_QWORD *)i[1];
        if ( !i )
          return;
      }
      ((void (__fastcall *)(_QWORD *, __int64 *))v3)(i, a2);
    }
  }
  else
  {
    sub_1D444E0(a1, (__int64)a2, (__int64)v4);
    for ( j = *(_QWORD **)(a1 + 664); j; j = (_QWORD *)j[1] )
      (*(void (__fastcall **)(_QWORD *, __int64 *, __int64 *))(*j + 16LL))(j, a2, v5);
    sub_1D182E0(a1, (__int64)a2);
  }
}
