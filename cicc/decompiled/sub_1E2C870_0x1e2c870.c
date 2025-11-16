// Function: sub_1E2C870
// Address: 0x1e2c870
//
void *__fastcall sub_1E2C870(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  __int64 v6; // r14
  __int64 v7; // rdi

  v3 = *(unsigned int *)(a1 + 1776);
  *(_QWORD *)a1 = &unk_49FBE00;
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 1760);
    v5 = &v4[2 * v3];
    do
    {
      if ( *v4 != -16 && *v4 != -8 )
      {
        v6 = v4[1];
        if ( v6 )
        {
          sub_1E11810(v4[1], a2);
          a2 = 752;
          j_j___libc_free_0(v6, 752);
        }
      }
      v4 += 2;
    }
    while ( v5 != v4 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1760));
  v7 = *(_QWORD *)(a1 + 1704);
  if ( v7 )
    j_j___libc_free_0(v7, *(_QWORD *)(a1 + 1720) - v7);
  sub_38C0FE0(a1 + 168);
  return sub_16367B0((_QWORD *)a1);
}
