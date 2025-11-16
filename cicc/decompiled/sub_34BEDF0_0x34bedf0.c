// Function: sub_34BEDF0
// Address: 0x34bedf0
//
__int64 __fastcall sub_34BEDF0(__int64 a1, char a2, char a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  __int64 result; // rax

  *(_QWORD *)(a1 + 32) = a1 + 56;
  *(_WORD *)(a1 + 128) = 0;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 132) = a7;
  *(_QWORD *)(a1 + 176) = a1 + 200;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 2;
  *(_DWORD *)(a1 + 48) = 0;
  *(_BYTE *)(a1 + 52) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 130) = a3;
  *(_BYTE *)(a1 + 131) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 8;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = a4;
  *(_QWORD *)(a1 + 240) = a5;
  result = (unsigned int)qword_503AD08;
  *(_QWORD *)(a1 + 248) = a6;
  if ( (_DWORD)result )
  {
    if ( (_DWORD)result == 1 )
      *(_BYTE *)(a1 + 129) = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 129) = a2;
  }
  return result;
}
