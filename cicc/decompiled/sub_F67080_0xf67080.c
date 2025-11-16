// Function: sub_F67080
// Address: 0xf67080
//
void __fastcall sub_F67080(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  __int64 v3; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_F67080(v1[3]);
      v3 = v1[6];
      v1 = (_QWORD *)v1[2];
      if ( v3 != -4096 && v3 != 0 && v3 != -8192 )
        sub_BD60C0(v2 + 4);
      j_j___libc_free_0(v2, 56);
    }
    while ( v1 );
  }
}
