// Function: sub_104C6D0
// Address: 0x104c6d0
//
void __fastcall sub_104C6D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r8
  __int64 v5; // rdx
  __int64 i; // rax

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(a3 + 40);
  if ( v4 == v5 )
  {
    if ( *(_BYTE *)a2 != 84 || *(_BYTE *)a3 != 84 )
    {
      for ( i = *(_QWORD *)(v4 + 56); !i || i - 24 != a2 && i - 24 != a3; i = *(_QWORD *)(i + 8) )
        ;
    }
  }
  else
  {
    sub_B19AA0(a1, *(_QWORD *)(a2 + 40), v5);
  }
}
