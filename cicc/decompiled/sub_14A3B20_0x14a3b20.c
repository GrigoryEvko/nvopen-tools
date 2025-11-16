// Function: sub_14A3B20
// Address: 0x14a3b20
//
void __fastcall sub_14A3B20(_QWORD *a1)
{
  _QWORD *v1; // r12
  __int64 (__fastcall *v2)(_QWORD *); // rax

  v1 = (_QWORD *)*a1;
  if ( *a1 )
  {
    v2 = *(__int64 (__fastcall **)(_QWORD *))(*v1 + 8LL);
    if ( v2 == sub_14A3AF0 )
    {
      *v1 = off_4984830;
      nullsub_542();
      j_j___libc_free_0(v1, 16);
    }
    else
    {
      v2((_QWORD *)*a1);
    }
  }
}
