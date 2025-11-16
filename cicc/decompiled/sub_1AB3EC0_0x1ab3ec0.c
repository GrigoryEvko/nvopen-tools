// Function: sub_1AB3EC0
// Address: 0x1ab3ec0
//
void __fastcall sub_1AB3EC0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax

  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 == *(_QWORD *)(a1 + 16) )
  {
    sub_1893930((unsigned __int64 **)a1, (char *)v3, a2);
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = 6;
      v4 = a2[2];
      *(_QWORD *)(v3 + 8) = 0;
      *(_QWORD *)(v3 + 16) = v4;
      if ( v4 != 0 && v4 != -8 && v4 != -16 )
        sub_1649AC0((unsigned __int64 *)v3, *a2 & 0xFFFFFFFFFFFFFFF8LL);
      v3 = *(_QWORD *)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v3 + 24;
  }
}
