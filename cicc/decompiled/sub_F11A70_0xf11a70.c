// Function: sub_F11A70
// Address: 0xf11a70
//
bool __fastcall sub_F11A70(__int64 a1, __int64 *a2)
{
  __int64 v3; // r9
  int v4; // ecx
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // r8
  int v8; // r11d
  unsigned __int64 v9; // rax
  unsigned int i; // eax
  _QWORD *v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rcx

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v3 = a1 + 16;
    v4 = 7;
    v5 = (_QWORD *)(a1 + 144);
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 16);
    v13 = *(unsigned int *)(a1 + 24);
    v5 = (_QWORD *)(v3 + 16 * v13);
    if ( !(_DWORD)v13 )
      return 0;
    v4 = v13 - 1;
  }
  v6 = *a2;
  v7 = a2[1];
  v8 = 1;
  v9 = 0xBF58476D1CE4E5B9LL
     * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
      | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32));
  for ( i = v4 & ((v9 >> 31) ^ v9); ; i = v4 & v12 )
  {
    v11 = (_QWORD *)(v3 + 16LL * i);
    if ( *v11 == v6 && v11[1] == v7 )
      break;
    if ( *v11 == -4096 && v11[1] == -4096 )
      return 0;
    v12 = v8 + i;
    ++v8;
  }
  return v11 != v5;
}
