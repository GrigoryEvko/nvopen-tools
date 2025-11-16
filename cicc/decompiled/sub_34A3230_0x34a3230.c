// Function: sub_34A3230
// Address: 0x34a3230
//
__int64 __fastcall sub_34A3230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rsi
  int v7; // r11d
  unsigned int i; // eax
  __int64 v9; // r8
  __int64 v10; // rdi
  _QWORD *v11; // rcx
  _QWORD *v12; // rdx
  int v13; // r11d
  __int64 result; // rax

  v5 = *(_QWORD *)a1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 192LL) )
    return sub_34A2FF0(a1, 1, a3, a4, a5);
  v7 = *(_DWORD *)(v5 + 196);
  for ( i = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4) + 1;
        v7 != i;
        *(_BYTE *)(v5 + v10 + 176) = *(_BYTE *)(v5 + v9 + 176) )
  {
    v9 = i;
    v10 = i++ - 1;
    v11 = (_QWORD *)(v5 + 16 * v9);
    v12 = (_QWORD *)(v5 + 16 * v10);
    *v12 = *v11;
    v12[1] = v11[1];
  }
  v13 = v7 - 1;
  *(_DWORD *)(v5 + 196) = v13;
  result = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(result + 8) = v13;
  return result;
}
