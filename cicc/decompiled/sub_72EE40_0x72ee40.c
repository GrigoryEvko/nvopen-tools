// Function: sub_72EE40
// Address: 0x72ee40
//
void __fastcall sub_72EE40(__int64 a1, unsigned __int8 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 i; // rcx

  *(_QWORD *)(a1 + 40) = a3;
  if ( (*(_BYTE *)(a3 - 8) & 1) == 0 )
  {
    v3 = *(int *)(a3 + 240);
    if ( (_DWORD)v3 == -1 )
    {
      i = qword_4F04C50;
      if ( !qword_4F04C50 )
      {
        for ( i = a3; *(_BYTE *)(i + 28) != 17; i = *(_QWORD *)(i + 16) )
        {
          i = *(_QWORD *)(i + 16);
          if ( *(_BYTE *)(i + 28) == 17 )
            break;
        }
      }
    }
    else
    {
      v4 = *(int *)(qword_4F04C68[0] + 776 * v3 + 400);
      if ( (_DWORD)v4 == -1 )
        return;
      i = *(_QWORD *)(qword_4F04C68[0] + 776 * v4 + 184);
    }
    *(_QWORD *)(a1 + 48) = *(_QWORD *)(i + 32);
    if ( (*(_BYTE *)(a1 - 8) & 1) != 0 )
    {
      sub_72EDB0(a3, a1, a2, i);
      *(_BYTE *)(a1 + 89) |= 2u;
      *(_QWORD *)(a1 + 40) = 0;
    }
  }
}
