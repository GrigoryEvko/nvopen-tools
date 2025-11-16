// Function: sub_734E20
// Address: 0x734e20
//
void __fastcall sub_734E20(__int64 a1)
{
  char v1; // r12
  __int64 v2; // rax
  __int64 i; // rbx

  if ( (*(_BYTE *)(a1 + 141) & 0x20) == 0 )
  {
    if ( *(char *)(a1 + 178) >= 0 )
    {
      if ( dword_4F077C4 == 2 )
        sub_734AF0(a1);
      sub_86ACD0(a1);
      v1 = *(_BYTE *)(*(_QWORD *)(a1 + 168) + 108LL);
      sub_7252E0(a1);
      *(_BYTE *)(*(_QWORD *)(a1 + 168) + 108LL) = v1;
    }
    else if ( dword_4F077C4 == 2 )
    {
      v2 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 152LL);
      if ( v2 )
      {
        if ( (*(_BYTE *)(v2 + 29) & 0x20) == 0 )
        {
          for ( i = *(_QWORD *)(v2 + 104); i; i = *(_QWORD *)(i + 112) )
          {
            if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u )
              sub_734E20(i);
          }
        }
      }
    }
  }
}
