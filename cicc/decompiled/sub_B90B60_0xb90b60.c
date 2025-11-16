// Function: sub_B90B60
// Address: 0xb90b60
//
__int64 __fastcall sub_B90B60(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // dl
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 *v5; // r9
  _QWORD *v6; // rax
  __int64 result; // rax

  v2 = *(_BYTE *)(a2 - 16);
  v3 = a2 - 16;
  v4 = a2;
  if ( *(_BYTE *)a2 != 16 )
  {
    if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
      v5 = *(__int64 **)(a2 - 32);
    else
      v5 = (__int64 *)(v3 - 8LL * ((v2 >> 2) & 0xF));
    v4 = *v5;
  }
  *(_QWORD *)a1 = v4;
  if ( (v2 & 2) != 0 )
  {
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 24LL);
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 32LL);
    v6 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    v6 = (_QWORD *)(v3 - 8LL * ((v2 >> 2) & 0xF));
    *(_QWORD *)(a1 + 8) = v6[1];
    *(_QWORD *)(a1 + 16) = v6[2];
    *(_QWORD *)(a1 + 24) = v6[3];
    *(_QWORD *)(a1 + 32) = v6[4];
  }
  *(_QWORD *)(a1 + 40) = v6[5];
  *(_DWORD *)(a1 + 48) = *(_DWORD *)(a2 + 4);
  result = *(_BYTE *)(a2 + 1) >> 7;
  *(_BYTE *)(a1 + 52) = result;
  return result;
}
