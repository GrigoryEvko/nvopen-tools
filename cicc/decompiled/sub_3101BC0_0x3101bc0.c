// Function: sub_3101BC0
// Address: 0x3101bc0
//
void __fastcall sub_3101BC0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi

  *(_QWORD *)a1 = off_4A32990;
  v2 = *(unsigned int *)(a1 + 32);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 16);
    v4 = &v3[7 * v2];
    do
    {
      if ( *v3 != -8192 && *v3 != -4096 )
      {
        v5 = v3[1];
        if ( (_QWORD *)v5 != v3 + 3 )
          _libc_free(v5);
      }
      v3 += 7;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 32);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 56 * v2, 8);
  nullsub_35();
}
