// Function: sub_FFE6D0
// Address: 0xffe6d0
//
__int64 __fastcall sub_FFE6D0(_BYTE *a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v6; // rbx
  __int64 v7; // r13
  unsigned int v8; // edi

  if ( *a1 != 86 )
    return 0;
  v4 = *((_QWORD *)a1 - 12);
  if ( (unsigned __int8)(*(_BYTE *)v4 - 82) > 1u )
    return 0;
  v6 = *(_QWORD *)(v4 - 64);
  v7 = *(_QWORD *)(v4 - 32);
  v8 = *(_WORD *)(v4 + 2) & 0x3F;
  if ( (v8 != a2 || v6 != a3 || v7 != a4) && ((unsigned int)sub_B52F50(v8) != a2 || v7 != a3 || v6 != a4) )
    return 0;
  return v4;
}
