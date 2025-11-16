// Function: sub_16F82E0
// Address: 0x16f82e0
//
_QWORD *__fastcall sub_16F82E0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  _QWORD *result; // rax
  __int64 v7; // r12

  v5 = **a1;
  result = *(_QWORD **)(v5 + 48);
  if ( *(_QWORD *)(v5 + 40) >= (unsigned __int64)result )
  {
    result = (_QWORD *)((char *)result - 1);
    *(_QWORD *)(v5 + 40) = result;
  }
  v7 = *(_QWORD *)(v5 + 344);
  if ( v7 )
  {
    result = (_QWORD *)sub_2241E50(a1, a2, a3, a4, a5);
    *(_DWORD *)v7 = 22;
    *(_QWORD *)(v7 + 8) = result;
  }
  if ( !*(_BYTE *)(v5 + 74) )
    result = sub_16D14E0(*(__int64 **)v5, *(_QWORD *)(v5 + 40), 0, a2, 0, 0, 0, 0, *(_BYTE *)(v5 + 75));
  *(_BYTE *)(v5 + 74) = 1;
  return result;
}
