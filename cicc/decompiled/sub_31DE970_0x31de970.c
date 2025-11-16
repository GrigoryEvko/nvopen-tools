// Function: sub_31DE970
// Address: 0x31de970
//
void __fastcall sub_31DE970(__int64 a1, __int64 a2, int a3, char a4)
{
  __int64 v4; // rax

  if ( a3 == 1 )
  {
    v4 = *(_QWORD *)(a1 + 208);
    if ( a4 )
    {
      if ( !*(_DWORD *)(v4 + 316) )
        return;
    }
    else if ( !*(_DWORD *)(v4 + 324) )
    {
      return;
    }
LABEL_4:
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 224) + 296LL))(*(_QWORD *)(a1 + 224));
    return;
  }
  if ( a3 == 2 && *(_DWORD *)(*(_QWORD *)(a1 + 208) + 328LL) )
    goto LABEL_4;
}
