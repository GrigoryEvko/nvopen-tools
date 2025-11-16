// Function: sub_7E9A00
// Address: 0x7e9a00
//
void __fastcall sub_7E9A00(__int64 a1)
{
  int v1; // r12d
  unsigned int v2; // r13d
  __int64 j; // rax
  __int64 v4; // rdx
  __int64 i; // rax
  __int64 v6; // rdx

  v1 = dword_4D03F8C;
  v2 = dword_4D03F8C == 0;
  sub_7DFEC0(*(_QWORD *)(a1 + 88), v2);
  if ( *(_BYTE *)(a1 + 28) == 17 )
  {
    sub_7DFEC0(*(_QWORD *)(a1 + 56), v2);
    if ( !v1 )
    {
      for ( i = *(_QWORD *)(a1 + 136); i; i = *(_QWORD *)(i + 112) )
      {
        v6 = *(_QWORD *)(i + 128);
        if ( v6 )
          *(_QWORD *)(v6 + 80) = 0;
      }
    }
  }
  else if ( !v1 )
  {
    for ( j = qword_4F06D10[42]; j; *(_QWORD *)(v4 - 16) = 0 )
    {
      v4 = j;
      j = *(_QWORD *)(j - 16);
    }
    qword_4F06D10[42] = 0;
    qword_4F06D10[43] = 0;
  }
}
