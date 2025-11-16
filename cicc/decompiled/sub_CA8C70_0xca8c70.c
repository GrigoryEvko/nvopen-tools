// Function: sub_CA8C70
// Address: 0xca8c70
//
_QWORD *__fastcall sub_CA8C70(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // r12
  __int64 v6; // rbx
  _QWORD *result; // rax
  __int64 v8; // r13

  v5 = *(_QWORD *)(a3 + 8);
  v6 = **a1;
  result = *(_QWORD **)(v6 + 48);
  v8 = *(_QWORD *)(v6 + 336);
  if ( v5 >= (unsigned __int64)result )
    v5 = (unsigned __int64)result - 1;
  if ( v8 )
  {
    result = (_QWORD *)sub_2241E50(a1, a2, (char *)result - 1, a4, a5);
    *(_DWORD *)v8 = 22;
    *(_QWORD *)(v8 + 8) = result;
  }
  if ( !*(_BYTE *)(v6 + 75) )
    result = sub_C91CB0(*(__int64 **)v6, v5, 0, a2, 0, 0, 0, 0, *(_BYTE *)(v6 + 76));
  *(_BYTE *)(v6 + 75) = 1;
  return result;
}
