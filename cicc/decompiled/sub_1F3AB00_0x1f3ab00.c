// Function: sub_1F3AB00
// Address: 0x1f3ab00
//
__int64 __fastcall sub_1F3AB00(__int64 a1, __int64 a2)
{
  __int16 v2; // ax
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 result; // rax
  int v6; // eax
  __int16 v7; // ax
  __int16 v8; // cx
  char v9; // dl
  __int64 v10; // rax
  __int64 (*v11)(); // rdx

  v2 = *(_WORD *)(a2 + 46);
  if ( (v2 & 4) == 0 && (v2 & 8) != 0 )
  {
    LOBYTE(v6) = sub_1E15D00(a2, 0x40u, 1);
    LODWORD(v4) = v6;
    if ( !(_BYTE)v6 )
      return (unsigned int)v4;
  }
  else
  {
    v3 = *(_QWORD *)(a2 + 16);
    v4 = (*(_QWORD *)(v3 + 8) >> 6) & 1LL;
    if ( (*(_QWORD *)(v3 + 8) & 0x40LL) == 0 )
      return (unsigned int)v4;
  }
  v7 = *(_WORD *)(a2 + 46);
  v8 = v7 & 4;
  if ( (v7 & 4) == 0 && (v7 & 8) != 0 )
  {
    v9 = sub_1E15D00(a2, 0x80u, 1);
    v7 = *(_WORD *)(a2 + 46);
    v8 = v7 & 4;
  }
  else
  {
    v9 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 7;
  }
  if ( v9 )
  {
    if ( v8 || (v7 & 8) == 0 )
      v10 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 5) & 1LL;
    else
      LOBYTE(v10) = sub_1E15D00(a2, 0x20u, 1);
    if ( !(_BYTE)v10 )
      return (unsigned int)v4;
    v7 = *(_WORD *)(a2 + 46);
    v8 = v7 & 4;
  }
  if ( !v8 && (v7 & 8) != 0 )
    LOBYTE(result) = sub_1E15D00(a2, 0x40000u, 2);
  else
    result = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 18) & 1LL;
  if ( !(_BYTE)result )
    return (unsigned int)v4;
  v11 = *(__int64 (**)())(*(_QWORD *)a1 + 656LL);
  if ( v11 != sub_1D918C0 )
    LODWORD(result) = ((__int64 (__fastcall *)(__int64, __int64))v11)(a1, a2) ^ 1;
  return (unsigned int)result;
}
