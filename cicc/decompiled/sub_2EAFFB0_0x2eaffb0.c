// Function: sub_2EAFFB0
// Address: 0x2eaffb0
//
__int64 __fastcall sub_2EAFFB0(__int64 a1)
{
  __int64 result; // rax

  *(_QWORD *)(a1 + 8) = a1 + 32;
  result = a1;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &unk_4F82428;
  *(_QWORD *)a1 = 1;
  if ( &unk_4F82428 != (_UNKNOWN *)&qword_4F82400 && &unk_4F82428 != &unk_4F82420 )
  {
    *(_DWORD *)(a1 + 20) = 2;
    *(_QWORD *)(a1 + 40) = &unk_4F82420;
    *(_QWORD *)a1 = 2;
  }
  return result;
}
