// Function: sub_6E62A0
// Address: 0x6e62a0
//
void __fastcall sub_6E62A0(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // r9
  int *v3; // rcx
  __int64 v4; // rax
  int v5; // [rsp+Ch] [rbp-14h] BYREF

  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned int)sub_6E6010() )
    {
      v3 = 0;
      v5 = 0;
      if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
        v3 = &v5;
      if ( dword_4F077C4 == 2 )
      {
        v4 = *(_QWORD *)(a1 + 24);
        if ( v4 )
        {
          if ( (*(_DWORD *)(v4 + 80) & 0x41000) != 0 )
          {
            sub_8841F0(a1, 0, 0, v3);
            if ( v5 )
              sub_6E50A0();
          }
        }
      }
    }
    else if ( (unsigned int)sub_87DC80(a1, 0, 0, *(char *)(qword_4D03C50 + 18LL) >= 0, v1, v2)
           && *(char *)(qword_4D03C50 + 18LL) < 0 )
    {
      sub_6E50A0();
    }
  }
}
