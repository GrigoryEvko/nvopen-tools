// Function: sub_E8EB90
// Address: 0xe8eb90
//
__int64 __fastcall sub_E8EB90(__int64 a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // rbx
  __int64 v3; // rax

  v1 = *(_QWORD **)(a1 + 8);
  v2 = &v1[5 * *(unsigned int *)(a1 + 16)];
  while ( v1 != v2 )
  {
    while ( 1 )
    {
      v2 -= 5;
      if ( (_QWORD *)*v2 == v2 + 2 )
        break;
      j_j___libc_free_0(*v2, v2[2] + 1LL);
      if ( v1 == v2 )
        goto LABEL_5;
    }
  }
LABEL_5:
  *(_DWORD *)(a1 + 16) = 0;
  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 != *(_QWORD *)(a1 + 64) )
    *(_QWORD *)(a1 + 64) = v3;
  *(_DWORD *)(a1 + 96) = 0;
  *(_WORD *)(a1 + 80) = 0;
  return 0;
}
