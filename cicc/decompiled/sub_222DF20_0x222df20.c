// Function: sub_222DF20
// Address: 0x222df20
//
int __fastcall sub_222DF20(__int64 a1)
{
  _QWORD *v1; // rax
  volatile signed __int32 **v2; // rdi

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)a1 = off_4A067E8;
  v1 = (_QWORD *)(a1 + 64);
  do
  {
    *v1 = 0;
    v1 += 2;
    *(v1 - 1) = 0;
  }
  while ( (_QWORD *)(a1 + 192) != v1 );
  *(_QWORD *)(a1 + 200) = a1 + 64;
  v2 = (volatile signed __int32 **)(a1 + 208);
  *((_DWORD *)v2 - 4) = 8;
  return sub_220A990(v2);
}
