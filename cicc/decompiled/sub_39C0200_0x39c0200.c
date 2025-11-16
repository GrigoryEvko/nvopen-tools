// Function: sub_39C0200
// Address: 0x39c0200
//
void __fastcall sub_39C0200(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax

  v2 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(*(_QWORD *)(v2 + 240) + 348LL) == 3 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(v2 + 256) + 16LL) + 80LL))(*(_QWORD *)(*(_QWORD *)(v2 + 256) + 16LL));
    v2 = *(_QWORD *)(a1 + 8);
  }
  if ( (unsigned int)sub_396EB00(v2) == 2 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    if ( !*(_BYTE *)(a1 + 25) )
    {
      if ( *(_BYTE *)(v3 + 536) )
      {
        (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(v3 + 256) + 712LL))(*(_QWORD *)(v3 + 256), 0, 1);
        v3 = *(_QWORD *)(a1 + 8);
      }
      *(_BYTE *)(a1 + 25) = 1;
    }
    *(_BYTE *)(a1 + 24) = 1;
    sub_38E0040(*(_QWORD *)(v3 + 256), 0);
  }
}
