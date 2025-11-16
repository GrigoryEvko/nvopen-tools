// Function: sub_B5B080
// Address: 0xb5b080
//
void __fastcall sub_B5B080(__int64 a1)
{
  _BYTE *v1; // rdi
  _WORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdi
  _BYTE *v7; // rdi

  v4 = *(_QWORD *)(a1 - 32);
  if ( !v4 || *(_BYTE *)v4 || (v5 = *(_QWORD *)(a1 + 80), *(_QWORD *)(v4 + 24) != v5) )
    BUG();
  v6 = *(_QWORD *)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  if ( *(_DWORD *)(v4 + 36) == 418 )
  {
    v1 = *(_BYTE **)(v6 + 24);
    if ( v1 )
    {
      if ( *v1 )
      {
        nullsub_2012();
      }
      else
      {
        v2 = (_WORD *)sub_B91420(v1, v5);
        if ( v3 == 3 )
        {
          switch ( *v2 )
          {
            case 0x656F:
              JUMPOUT(0xB58A50);
            case 0x676F:
              JUMPOUT(0xB58A38);
            case 0x6C6F:
              JUMPOUT(0xB58A08);
            case 0x6E6F:
              JUMPOUT(0xB589D8);
            case 0x726F:
              JUMPOUT(0xB589C0);
            case 0x6E75:
              JUMPOUT(0xB589A8);
          }
          switch ( *v2 )
          {
            case 0x6575:
              JUMPOUT(0xB58990);
            case 0x6775:
              JUMPOUT(0xB58980);
            case 0x6C75:
              JUMPOUT(0xB58960);
            case 0x6E75:
              JUMPOUT(0xB58940);
          }
        }
      }
    }
    else
    {
      nullsub_2011();
    }
  }
  else
  {
    v7 = *(_BYTE **)(v6 + 24);
    if ( v7 )
    {
      if ( !*v7 )
        sub_B91420(v7, v5);
    }
  }
}
