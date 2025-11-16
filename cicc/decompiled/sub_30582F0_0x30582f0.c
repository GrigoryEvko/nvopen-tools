// Function: sub_30582F0
// Address: 0x30582f0
//
void __fastcall sub_30582F0(unsigned __int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rbx

  *(_QWORD *)a1 = &unk_4A2F1C0;
  if ( *(void **)(a1 + 32) == sub_C33340() )
  {
    v1 = *(_QWORD **)(a1 + 40);
    if ( v1 )
    {
      v2 = &v1[3 * *(v1 - 1)];
      if ( v1 != v2 )
      {
        do
        {
          v2 -= 3;
          sub_91D830(v2);
        }
        while ( *(_QWORD **)(a1 + 40) != v2 );
      }
      j_j_j___libc_free_0_0((unsigned __int64)(v2 - 1));
    }
  }
  else
  {
    sub_C338F0(a1 + 32);
  }
  j_j___libc_free_0(a1);
}
