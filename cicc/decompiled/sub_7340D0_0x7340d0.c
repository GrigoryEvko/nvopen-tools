// Function: sub_7340D0
// Address: 0x7340d0
//
void __fastcall sub_7340D0(__int64 a1, int a2, int a3)
{
  _QWORD *v3; // r14
  unsigned __int8 *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax

  if ( *(_QWORD *)(a1 + 16) )
  {
    if ( dword_4D03F94
      || dword_4F04C64 == -1
      || dword_4F04C44 == -1
      && (v7 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v7 + 6) & 6) == 0)
      && *(_BYTE *)(v7 + 4) != 12 )
    {
      if ( a2 )
      {
        v3 = (_QWORD *)qword_4F04C50;
        if ( qword_4F04C50 )
        {
          v4 = *(unsigned __int8 **)(qword_4F04C50 + 56LL);
          if ( !v4 )
          {
            v4 = sub_726C30(3);
            sub_732E60(v4, 0x17u, v3);
          }
        }
        else
        {
          v4 = *(unsigned __int8 **)(qword_4F04C68[0] + 488LL);
        }
      }
      else
      {
        v4 = (unsigned __int8 *)qword_4F06BC0;
        if ( a3 )
          v4 = (unsigned __int8 *)sub_7340A0(qword_4F06BC0);
        v5 = *(_QWORD *)(a1 + 40);
        if ( !v5 )
          v5 = qword_4F06BC0;
        if ( v4 != (unsigned __int8 *)v5 )
        {
          do
          {
            v6 = v5;
            v5 = *(_QWORD *)(v5 + 32);
          }
          while ( (unsigned __int8 *)v5 != v4 );
          if ( *(_QWORD *)(v6 + 24) )
          {
            *(_QWORD *)(v6 + 40) = a1;
            *(_QWORD *)(a1 + 104) = v6;
            *(_BYTE *)(a1 + 50) |= 4u;
          }
        }
      }
      sub_732D90(a1, (__int64)v4);
    }
  }
}
