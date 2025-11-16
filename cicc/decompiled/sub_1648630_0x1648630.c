// Function: sub_1648630
// Address: 0x1648630
//
void __fastcall sub_1648630(_QWORD *a1, _QWORD *a2, char a3)
{
  __int64 v4; // rdx
  unsigned __int64 v5; // rax

  while ( a1 != a2 )
  {
    a2 -= 3;
    if ( *a2 )
    {
      v4 = a2[1];
      v5 = a2[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v5 = v4;
      if ( v4 )
        *(_QWORD *)(v4 + 16) = *(_QWORD *)(v4 + 16) & 3LL | v5;
    }
  }
  if ( a3 )
    j___libc_free_0(a1);
}
