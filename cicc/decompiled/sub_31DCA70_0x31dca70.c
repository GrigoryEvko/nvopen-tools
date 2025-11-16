// Function: sub_31DCA70
// Address: 0x31dca70
//
void __fastcall sub_31DCA70(__int64 a1, unsigned int a2, __int64 a3, unsigned int a4)
{
  unsigned int v5; // r12d
  __int64 v7; // rax
  __int64 v8; // rdx

  v5 = a2;
  if ( a3 )
  {
    v7 = sub_B2F730(a3);
    v5 = sub_31DA250(a3, v7, a2);
  }
  if ( (_BYTE)v5 )
  {
    if ( (*(_BYTE *)(sub_31DB4F0(a1) + 48) & 0x10) != 0 )
    {
      if ( *(_QWORD *)(a1 + 232) )
        v8 = sub_31DB000(a1);
      else
        v8 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 680LL);
      (*(void (__fastcall **)(_QWORD, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 616LL))(
        *(_QWORD *)(a1 + 224),
        v5,
        v8,
        a4);
    }
    else
    {
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 608LL))(
        *(_QWORD *)(a1 + 224),
        v5,
        0,
        1,
        a4);
    }
  }
}
