// Function: sub_27EC260
// Address: 0x27ec260
//
void __fastcall sub_27EC260(_QWORD *a1)
{
  _QWORD *v1; // rbx
  unsigned __int64 v2; // r12
  __int64 v3; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_27EC260(v1[3]);
      v3 = v1[6];
      v1 = (_QWORD *)v1[2];
      if ( v3 != -4096 && v3 != 0 && v3 != -8192 )
        sub_BD60C0((_QWORD *)(v2 + 32));
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
