// Function: sub_1BE2750
// Address: 0x1be2750
//
__int64 __fastcall sub_1BE2750(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax

  if ( *a2 == 2 )
  {
    sub_1BE2400((__int64)(a2 - 40), a1);
    return a1;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 16) - v3) <= 2 )
    {
      v4 = sub_16E7EE0(a1, "%vp", 3u);
      sub_16E7AB0(v4, (unsigned __int16)a2);
    }
    else
    {
      *(_BYTE *)(v3 + 2) = 112;
      *(_WORD *)v3 = 30245;
      *(_QWORD *)(a1 + 24) += 3LL;
      sub_16E7AB0(a1, (unsigned __int16)a2);
    }
    return a1;
  }
}
