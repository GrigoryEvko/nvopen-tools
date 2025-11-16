// Function: sub_BE0AB0
// Address: 0xbe0ab0
//
__int64 __fastcall sub_BE0AB0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax

  v4 = sub_BC0510(a4, &unk_4F836C8, a3);
  if ( *a2 && (*(_BYTE *)(v4 + 8) || *(_BYTE *)(v4 + 9)) )
    sub_C64ED0("Broken module found, compilation aborted!", 1);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
