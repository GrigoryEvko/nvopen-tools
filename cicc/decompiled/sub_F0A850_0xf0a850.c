// Function: sub_F0A850
// Address: 0xf0a850
//
_QWORD *__fastcall sub_F0A850(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  int v5; // eax
  unsigned int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  _QWORD *result; // rax

  v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( *(_DWORD *)(a1 + 72) == v4 )
  {
    sub_B48D90(a1);
    v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  }
  v5 = (v4 + 1) & 0x7FFFFFF;
  v6 = v5 | *(_DWORD *)(a1 + 4) & 0xF8000000;
  v7 = *(_QWORD *)(a1 - 8) + 32LL * (unsigned int)(v5 - 1);
  *(_DWORD *)(a1 + 4) = v6;
  if ( *(_QWORD *)v7 )
  {
    v8 = *(_QWORD *)(v7 + 8);
    **(_QWORD **)(v7 + 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v7 + 16);
  }
  *(_QWORD *)v7 = a2;
  if ( a2 )
  {
    v9 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v7 + 8) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = v7 + 8;
    *(_QWORD *)(v7 + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v7;
  }
  result = (_QWORD *)(*(_QWORD *)(a1 - 8)
                    + 32LL * *(unsigned int *)(a1 + 72)
                    + 8LL * ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) - 1));
  *result = a3;
  return result;
}
