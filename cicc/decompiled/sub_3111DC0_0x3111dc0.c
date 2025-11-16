// Function: sub_3111DC0
// Address: 0x3111dc0
//
__int64 __fastcall sub_3111DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v6; // edx
  int v7; // edi
  __int64 v8; // rcx
  char v9; // al
  int v11; // esi
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-18h] BYREF

  if ( *(_QWORD *)a2 != 0x81617461646763FFLL )
  {
    v11 = 2;
LABEL_7:
    sub_3111350(&v13, v11);
    v12 = v13;
    *(_BYTE *)(a1 + 32) |= 3u;
    *(_QWORD *)a1 = v12 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  v6 = *(_DWORD *)(a2 + 8);
  if ( v6 > 2 )
  {
    v11 = 6;
    goto LABEL_7;
  }
  v7 = *(_DWORD *)(a2 + 12);
  v8 = *(_QWORD *)(a2 + 16);
  if ( v6 == 2 )
    a5 = *(_QWORD *)(a2 + 24);
  v9 = *(_BYTE *)(a1 + 32);
  *(_DWORD *)(a1 + 8) = v6;
  *(_DWORD *)(a1 + 12) = v7;
  *(_QWORD *)(a1 + 16) = v8;
  *(_QWORD *)(a1 + 24) = a5;
  *(_BYTE *)(a1 + 32) = v9 & 0xFC | 2;
  *(_QWORD *)a1 = 0x81617461646763FFLL;
  return a1;
}
