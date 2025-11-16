// Function: sub_FFBD40
// Address: 0xffbd40
//
void __fastcall sub_FFBD40(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rax

  if ( *(_BYTE *)(a1 + 560) )
  {
    *(_WORD *)(a1 + 664) = 257;
    sub_FFBA00(a1, a2);
    v5 = *(_QWORD *)(a1 + 544);
    if ( v5 )
    {
      *(_QWORD *)(v5 + 104) = a2;
      *(_DWORD *)(v5 + 120) = *(_DWORD *)(a2 + 92);
      sub_B1F440(v5);
    }
    v6 = *(_QWORD *)(a1 + 552);
    if ( v6 )
    {
      *(_QWORD *)(v6 + 128) = a2;
      *(_DWORD *)(v6 + 144) = *(_DWORD *)(a2 + 92);
      sub_B29120(v6);
    }
    *(_WORD *)(a1 + 664) = 0;
    v7 = *(unsigned int *)(a1 + 8);
    *(_QWORD *)(a1 + 536) = v7;
    *(_QWORD *)(a1 + 528) = v7;
    sub_FFBC40(a1, a2);
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 544);
    if ( v3 )
    {
      *(_QWORD *)(v3 + 104) = a2;
      *(_DWORD *)(v3 + 120) = *(_DWORD *)(a2 + 92);
      sub_B1F440(v3);
    }
    v4 = *(_QWORD *)(a1 + 552);
    if ( v4 )
    {
      *(_QWORD *)(v4 + 128) = a2;
      *(_DWORD *)(v4 + 144) = *(_DWORD *)(a2 + 92);
      sub_B29120(v4);
    }
  }
}
