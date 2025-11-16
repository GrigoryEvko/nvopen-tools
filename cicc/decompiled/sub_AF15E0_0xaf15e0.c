// Function: sub_AF15E0
// Address: 0xaf15e0
//
__int64 __fastcall sub_AF15E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 v3; // dl
  __int64 v4; // rcx
  int v5; // edx
  __int64 result; // rax
  __int64 v7; // rax

  v2 = sub_B10CD0(a2 + 48);
  v3 = *(_BYTE *)(v2 - 16);
  if ( (v3 & 2) == 0 )
  {
    if ( ((*(_WORD *)(v2 - 16) >> 6) & 0xF) != 2 )
      goto LABEL_3;
    v7 = v2 - 16 - 8LL * ((v3 >> 2) & 0xF);
LABEL_6:
    v4 = *(_QWORD *)(v7 + 8);
    goto LABEL_4;
  }
  if ( *(_DWORD *)(v2 - 24) == 2 )
  {
    v7 = *(_QWORD *)(v2 - 32);
    goto LABEL_6;
  }
LABEL_3:
  v4 = 0;
LABEL_4:
  v5 = *(_DWORD *)(a2 + 4);
  *(_BYTE *)(a1 + 24) = 0;
  result = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1LL - (v5 & 0x7FFFFFF))) + 24LL);
  *(_QWORD *)(a1 + 32) = v4;
  *(_QWORD *)a1 = result;
  return result;
}
