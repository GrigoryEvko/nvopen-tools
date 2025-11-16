// Function: sub_38C5720
// Address: 0x38c5720
//
__int64 __fastcall sub_38C5720(_QWORD *a1, __int64 a2, char a3, char a4, __int64 a5)
{
  bool v9; // zf
  __int64 v10; // r8
  _QWORD *v11; // rsi
  __int64 result; // rax
  char v13; // al
  __int64 v14; // rcx
  _QWORD *v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rsi

  v9 = *(_BYTE *)(a5 + 72) == 0;
  v10 = *(_QWORD *)(a2 + 8);
  v11 = *(_QWORD **)a2;
  if ( v9 )
  {
    (*(void (__fastcall **)(_QWORD *, _QWORD *, __int64))(*a1 + 400LL))(a1, v11, v10);
    (*(void (__fastcall **)(_QWORD *, void *, __int64))(*a1 + 400LL))(a1, &unk_452DFBC, 1);
  }
  else
  {
    sub_38C5650(a5, a1, v11, v10);
  }
  result = sub_38DCDD0(a1, *(unsigned int *)(a2 + 32));
  if ( a3 )
    result = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64))(*a1 + 408LL))(a1, *(_QWORD *)(a2 + 40), 16);
  if ( a4 )
  {
    v13 = *(_BYTE *)(a2 + 64);
    if ( *(_BYTE *)(a5 + 72) )
    {
      if ( v13 )
      {
        v15 = *(_QWORD **)(a2 + 48);
        v14 = *(_QWORD *)(a2 + 56);
      }
      else
      {
        v14 = 0;
        v15 = 0;
      }
      return sub_38C5650(a5, a1, v15, v14);
    }
    else
    {
      if ( v13 )
      {
        v17 = *(_QWORD *)(a2 + 48);
        v16 = *(_QWORD *)(a2 + 56);
      }
      else
      {
        v16 = 0;
        v17 = 0;
      }
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 400LL))(a1, v17, v16);
      return (*(__int64 (__fastcall **)(_QWORD *, void *, __int64))(*a1 + 400LL))(a1, &unk_452DFBC, 1);
    }
  }
  return result;
}
