// Function: sub_253C2D0
// Address: 0x253c2d0
//
__int64 __fastcall sub_253C2D0(__int64 a1)
{
  char v2; // al
  __int64 v3; // rsi
  __int64 v4; // rdi
  void *v6; // rax
  __int64 v7; // rdx
  const void *v8; // rsi

  v2 = *(_BYTE *)(a1 + 48);
  v3 = *(unsigned int *)(a1 + 40);
  *(_BYTE *)(a1 + 88) = 1;
  *(_BYTE *)(a1 + 8) = v2;
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 16 * v3, 8);
  v4 = *(unsigned int *)(a1 + 80);
  *(_DWORD *)(a1 + 40) = v4;
  if ( (_DWORD)v4 )
  {
    v6 = (void *)sub_C7D670(16 * v4, 8);
    v7 = *(unsigned int *)(a1 + 40);
    v8 = *(const void **)(a1 + 64);
    *(_QWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(a1 + 72);
    memcpy(v6, v8, 16 * v7);
  }
  else
  {
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 32) = 0;
  }
  return 1;
}
