// Function: sub_E4CD30
// Address: 0xe4cd30
//
void __fastcall sub_E4CD30(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax

  if ( *(_BYTE *)(a1 + 745) )
  {
    sub_CA0EC0(a2, a1 + 488);
    if ( a3 )
    {
      v6 = *(_QWORD *)(a1 + 496);
      if ( (unsigned __int64)(v6 + 1) > *(_QWORD *)(a1 + 504) )
      {
        sub_C8D290(a1 + 488, (const void *)(a1 + 512), v6 + 1, 1u, v4, v5);
        v6 = *(_QWORD *)(a1 + 496);
      }
      *(_BYTE *)(*(_QWORD *)(a1 + 488) + v6) = 10;
      ++*(_QWORD *)(a1 + 496);
    }
  }
}
