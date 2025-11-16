// Function: sub_3982460
// Address: 0x3982460
//
void __fastcall sub_3982460(__int64 *a1, __int64 a2, unsigned __int16 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // [rsp+8h] [rbp-8h] BYREF

  v6 = *a1;
  if ( a3 == 14 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 240) + 356LL) )
    {
      v7 = *(_QWORD *)(v6 + 8);
      sub_3982300(&v7, a2, 14, a4, a5, a6);
    }
    else
    {
      v7 = *(unsigned int *)(v6 + 16);
      sub_39820D0(&v7, a2, 0xEu);
    }
  }
  else
  {
    v7 = *(unsigned int *)(v6 + 20);
    sub_39820D0(&v7, a2, a3);
  }
}
