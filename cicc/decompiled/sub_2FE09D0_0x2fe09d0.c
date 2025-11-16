// Function: sub_2FE09D0
// Address: 0x2fe09d0
//
__int64 __fastcall sub_2FE09D0(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  int v7; // eax
  __int64 v8; // rdx
  __int64 (*v9)(); // rax

  v3 = *(unsigned __int16 *)(a3 + 68);
  if ( !(_WORD)v3 )
    return 0;
  if ( (unsigned __int16)(v3 - 9) <= 0x3Bu )
  {
    v5 = 0x800000000000C09LL;
    if ( _bittest64(&v5, (unsigned int)(v3 - 9)) )
      return 0;
  }
  v6 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 24LL);
  if ( (v6 & 0x10) != 0 )
    return 0;
  if ( (unsigned __int16)(v3 - 1) <= 1u && (*(_BYTE *)(*(_QWORD *)(a3 + 32) + 64LL) & 8) != 0 )
    return *(unsigned int *)(a2 + 12);
  v7 = *(_DWORD *)(a3 + 44);
  if ( (v7 & 4) != 0 || (v7 & 8) == 0 )
    v8 = (v6 >> 19) & 1;
  else
    LOBYTE(v8) = sub_2E88A90(a3, 0x80000, 1);
  if ( (_BYTE)v8 )
    return *(unsigned int *)(a2 + 12);
  v9 = *(__int64 (**)())(*(_QWORD *)a1 + 1184LL);
  if ( v9 == sub_2FDC750 || !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v9)(a1, *(unsigned __int16 *)(a3 + 68)) )
    return 1;
  else
    return *(unsigned int *)(a2 + 16);
}
