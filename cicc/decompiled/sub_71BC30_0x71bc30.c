// Function: sub_71BC30
// Address: 0x71bc30
//
void __fastcall sub_71BC30(__int64 a1)
{
  __int64 i; // r12
  __int64 **j; // r14
  __int64 *k; // r13
  _QWORD *v4; // rbx
  __int64 m; // rax

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (*(_BYTE *)(i + 177) & 1) != 0 )
  {
    sub_71C980(i);
    if ( (dword_4F077BC && qword_4F077A8 <= 0x9FC3u || (_DWORD)qword_4F077B4) && (*(_BYTE *)(i + 176) & 0x10) != 0 )
    {
      for ( j = **(__int64 ****)(i + 168); j; j = (__int64 **)*j )
      {
        if ( ((_BYTE)j[12] & 2) != 0 )
        {
          for ( k = j[14]; k; k = (__int64 *)*k )
          {
            v4 = (_QWORD *)k[1];
            for ( m = v4[2]; (*(_BYTE *)(m + 96) & 2) == 0; m = v4[2] )
            {
              sub_71C980(*(_QWORD *)(m + 40));
              v4 = (_QWORD *)*v4;
            }
          }
        }
      }
    }
    sub_71AF80(i);
  }
}
