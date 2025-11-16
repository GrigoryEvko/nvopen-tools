// Function: sub_2EAC0D0
// Address: 0x2eac0d0
//
void __fastcall sub_2EAC0D0(__int64 a1, signed __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // rax

  if ( a2 )
  {
    v2 = *(_QWORD *)(a1 + 32);
    v3 = *(_QWORD *)(a1 + 24) - v2;
    if ( a2 < 0 )
    {
      if ( v3 <= 2 )
      {
        a1 = sub_CB6200(a1, (unsigned __int8 *)" - ", 3u);
      }
      else
      {
        *(_BYTE *)(v2 + 2) = 32;
        *(_WORD *)v2 = 11552;
        *(_QWORD *)(a1 + 32) += 3LL;
      }
      sub_CB59F0(a1, -a2);
    }
    else if ( v3 <= 2 )
    {
      v4 = sub_CB6200(a1, " + ", 3u);
      sub_CB59F0(v4, a2);
    }
    else
    {
      *(_BYTE *)(v2 + 2) = 32;
      *(_WORD *)v2 = 11040;
      *(_QWORD *)(a1 + 32) += 3LL;
      sub_CB59F0(a1, a2);
    }
  }
}
