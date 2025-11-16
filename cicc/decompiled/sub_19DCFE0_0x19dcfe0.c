// Function: sub_19DCFE0
// Address: 0x19dcfe0
//
void *__fastcall sub_19DCFE0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  __int64 v3; // rbx
  _QWORD *v4; // r15
  _QWORD *v5; // r12
  __int64 v6; // rax

  *(_QWORD *)a1 = off_49F4BA8;
  v1 = *(unsigned int *)(a1 + 232);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD *)(a1 + 216);
    v3 = v2 + 72 * v1;
    do
    {
      if ( *(_QWORD *)v2 != -16 && *(_QWORD *)v2 != -8 )
      {
        v4 = *(_QWORD **)(v2 + 8);
        v5 = &v4[3 * *(unsigned int *)(v2 + 16)];
        if ( v4 != v5 )
        {
          do
          {
            v6 = *(v5 - 1);
            v5 -= 3;
            if ( v6 != -8 && v6 != 0 && v6 != -16 )
              sub_1649B30(v5);
          }
          while ( v4 != v5 );
          v5 = *(_QWORD **)(v2 + 8);
        }
        if ( v5 != (_QWORD *)(v2 + 24) )
          _libc_free((unsigned __int64)v5);
      }
      v2 += 72;
    }
    while ( v3 != v2 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 216));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
