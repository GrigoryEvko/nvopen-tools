// Function: sub_1E9C2C0
// Address: 0x1e9c2c0
//
__int64 __fastcall sub_1E9C2C0(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rcx
  _DWORD *v6; // rcx

  result = 0;
  if ( !*(_DWORD *)(a1 + 16) )
  {
    v4 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 16) = 1;
    v5 = *(_QWORD *)(v4 + 32);
    LODWORD(v4) = *(_DWORD *)(v5 + 40);
    *a2 = *(_DWORD *)(v5 + 48);
    a2[1] = ((unsigned int)v4 >> 8) & 0xFFF;
    v6 = *(_DWORD **)(*(_QWORD *)(a1 + 8) + 32LL);
    LODWORD(v4) = *v6;
    *a3 = v6[2];
    a3[1] = ((unsigned int)v4 >> 8) & 0xFFF;
    return 1;
  }
  return result;
}
