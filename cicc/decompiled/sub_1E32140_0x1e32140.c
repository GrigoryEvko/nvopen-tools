// Function: sub_1E32140
// Address: 0x1e32140
//
void __fastcall sub_1E32140(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // rax

  if ( a2 )
  {
    v2 = *(_QWORD *)(a1 + 24);
    v3 = *(_QWORD *)(a1 + 16) - v2;
    if ( a2 < 0 )
    {
      if ( v3 <= 2 )
      {
        a1 = sub_16E7EE0(a1, " - ", 3u);
      }
      else
      {
        *(_BYTE *)(v2 + 2) = 32;
        *(_WORD *)v2 = 11552;
        *(_QWORD *)(a1 + 24) += 3LL;
      }
      sub_16E7AB0(a1, -a2);
    }
    else if ( v3 <= 2 )
    {
      v4 = sub_16E7EE0(a1, " + ", 3u);
      sub_16E7AB0(v4, a2);
    }
    else
    {
      *(_BYTE *)(v2 + 2) = 32;
      *(_WORD *)v2 = 11040;
      *(_QWORD *)(a1 + 24) += 3LL;
      sub_16E7AB0(a1, a2);
    }
  }
}
