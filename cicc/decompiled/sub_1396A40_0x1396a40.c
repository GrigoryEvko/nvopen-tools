// Function: sub_1396A40
// Address: 0x1396a40
//
void __fastcall sub_1396A40(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r13
  _QWORD *v3; // r15
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v6; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1396A40(v1[3]);
      v3 = (_QWORD *)v1[5];
      v1 = (_QWORD *)v1[2];
      if ( v3 )
      {
        v4 = v3[2];
        v5 = v3[1];
        if ( v4 != v5 )
        {
          do
          {
            v6 = *(_QWORD *)(v5 + 16);
            if ( v6 != 0 && v6 != -8 && v6 != -16 )
              sub_1649B30(v5);
            v5 += 32;
          }
          while ( v4 != v5 );
          v5 = v3[1];
        }
        if ( v5 )
          j_j___libc_free_0(v5, v3[3] - v5);
        j_j___libc_free_0(v3, 40);
      }
      j_j___libc_free_0(v2, 48);
    }
    while ( v1 );
  }
}
