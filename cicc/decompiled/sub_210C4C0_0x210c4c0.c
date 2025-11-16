// Function: sub_210C4C0
// Address: 0x210c4c0
//
__int64 __fastcall sub_210C4C0(__int64 a1)
{
  __int64 v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rdi

  v2 = *(unsigned int *)(a1 + 184);
  *(_QWORD *)a1 = &unk_4A00ED0;
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 168);
    v4 = &v3[4 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[1];
        if ( v5 )
          j_j___libc_free_0(v5, v3[3] - v5);
      }
      v3 += 4;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 168));
  sub_16367B0((_QWORD *)a1);
  return j_j___libc_free_0(a1, 200);
}
