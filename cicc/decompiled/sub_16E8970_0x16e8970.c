// Function: sub_16E8970
// Address: 0x16e8970
//
__off_t __fastcall sub_16E8970(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5)
{
  __int64 v5; // rcx
  __off_t result; // rax
  bool v7; // zf

  v5 = a4 ^ 1u;
  *(_DWORD *)(a1 + 32) = v5;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = &unk_49EFB70;
  *(_DWORD *)(a1 + 36) = a2;
  *(_BYTE *)(a1 + 40) = a3;
  *(_DWORD *)(a1 + 48) = 0;
  result = sub_2241E40(a1, a2, a3, v5, a5);
  *(_QWORD *)(a1 + 56) = result;
  if ( (int)a2 < 0 )
  {
    *(_BYTE *)(a1 + 40) = 0;
  }
  else
  {
    if ( (int)a2 <= 2 )
      *(_BYTE *)(a1 + 40) = 0;
    result = lseek(a2, 0, 1);
    v7 = result == -1;
    if ( result == -1 )
      result = 0;
    *(_BYTE *)(a1 + 72) = !v7;
    *(_QWORD *)(a1 + 64) = result;
  }
  return result;
}
