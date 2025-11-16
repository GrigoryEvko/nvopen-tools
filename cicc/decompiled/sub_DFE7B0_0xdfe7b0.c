// Function: sub_DFE7B0
// Address: 0xdfe7b0
//
void __fastcall sub_DFE7B0(_QWORD *a1)
{
  _QWORD *v1; // r12
  __int64 (__fastcall *v2)(_QWORD *); // rax

  v1 = (_QWORD *)*a1;
  if ( *a1 )
  {
    v2 = *(__int64 (__fastcall **)(_QWORD *))(*v1 + 8LL);
    if ( v2 == sub_DFE780 )
    {
      *v1 = off_4979D10;
      nullsub_197();
      j_j___libc_free_0(v1, 16);
    }
    else
    {
      v2((_QWORD *)*a1);
    }
  }
}
