// Function: sub_234A9E0
// Address: 0x234a9e0
//
_QWORD *__fastcall sub_234A9E0(_QWORD *a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // r14
  _QWORD *v3; // r12
  unsigned __int64 v4; // r15
  _QWORD *v5; // rax
  _QWORD *v7; // rbx

  v2 = *a2;
  v3 = (_QWORD *)a2[1];
  *a2 = 0;
  v4 = a2[2];
  a2[1] = 0;
  a2[2] = 0;
  v5 = (_QWORD *)sub_22077B0(0x30u);
  if ( v5 )
  {
    v5[1] = v2;
    v5[2] = v3;
    v5[3] = v4;
    *v5 = &unk_4A0C3B8;
    v5[4] = 0;
    v5[5] = 0;
    *a1 = v5;
  }
  else
  {
    *a1 = 0;
    if ( (_QWORD *)v2 != v3 )
    {
      v7 = (_QWORD *)v2;
      do
      {
        if ( *v7 )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v7 + 8LL))(*v7);
        ++v7;
      }
      while ( v3 != v7 );
    }
    if ( v2 )
      j_j___libc_free_0(v2);
  }
  return a1;
}
