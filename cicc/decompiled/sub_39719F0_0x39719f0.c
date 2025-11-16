// Function: sub_39719F0
// Address: 0x39719f0
//
void __fastcall sub_39719F0(__int64 a1, __int64 a2, int a3, char a4)
{
  __int64 v4; // rax

  if ( a3 == 1 )
  {
    v4 = *(_QWORD *)(a1 + 240);
    if ( a4 )
    {
      if ( !*(_DWORD *)(v4 + 332) )
        return;
    }
    else if ( !*(_DWORD *)(v4 + 336) )
    {
      return;
    }
LABEL_4:
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 256) + 256LL))(*(_QWORD *)(a1 + 256));
    return;
  }
  if ( a3 == 2 && *(_DWORD *)(*(_QWORD *)(a1 + 240) + 340LL) )
    goto LABEL_4;
}
