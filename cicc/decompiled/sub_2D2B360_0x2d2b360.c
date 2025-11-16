// Function: sub_2D2B360
// Address: 0x2d2b360
//
_DWORD *__fastcall sub_2D2B360(__int64 a1, __int64 a2)
{
  int v2; // r13d
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rsi
  _DWORD *result; // rax
  int v8; // r13d
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9

  v2 = a2;
  v3 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v4 = *(_QWORD *)v3;
  v5 = *(unsigned int *)(v3 + 12);
  if ( (unsigned int)a2 >= *(_DWORD *)(v4 + 8 * v5 + 4) && sub_2D28D20(a1, a2, *(_DWORD *)(v4 + 4 * v5 + 128)) )
  {
    v8 = *(_DWORD *)sub_2D289F0(a1);
    sub_2D2B2D0(a1, a2, v9, v10, v11, v12);
    result = (_DWORD *)sub_2D289F0(a1);
    *result = v8;
  }
  else
  {
    *(_DWORD *)sub_2D28A10(a1) = a2;
    v6 = (unsigned int)(*(_DWORD *)(a1 + 16) - 1);
    result = (_DWORD *)(*(_QWORD *)(a1 + 8) + 16 * v6);
    if ( result[2] - 1 == result[3] && *(_DWORD *)(a1 + 16) != 1 )
      return (_DWORD *)sub_2D22A70(a1, v6, v2);
  }
  return result;
}
