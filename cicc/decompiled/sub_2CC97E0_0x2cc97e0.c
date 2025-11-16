// Function: sub_2CC97E0
// Address: 0x2cc97e0
//
char __fastcall sub_2CC97E0(__int64 a1, __int64 a2, int a3, __int64 a4, _BYTE *a5)
{
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v10; // rax
  int v11; // r13d
  _BYTE *v12; // rdi

  if ( *(_BYTE *)a2 != 62 )
  {
    if ( *(_BYTE *)a2 != 85 )
      return 0;
    v10 = *(_QWORD *)(a2 - 32);
    if ( v10 )
    {
      if ( !*(_BYTE *)v10 && *(_QWORD *)(v10 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v10 + 33) & 0x20) != 0 )
      {
        v11 = *(_DWORD *)(v10 + 36);
        if ( sub_CEA1A0(v11) || (unsigned __int8)sub_CEA1F0(v11) )
          return a3 != 5;
      }
    }
    if ( sub_B49E00(a2) || (unsigned __int8)sub_B49E20(a2) )
      return 0;
    v12 = *(_BYTE **)(a2 - 32);
    if ( *v12 == 25 )
      return sub_CF0FA0((__int64)v12);
    return 1;
  }
  v6 = *(_QWORD **)a4;
  v7 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  if ( *(_QWORD *)a4 != v7 )
  {
    do
    {
      if ( *v6 == a2 )
        *a5 = 1;
      ++v6;
    }
    while ( (_QWORD *)v7 != v6 );
  }
  if ( !a3 )
    return 1;
  v8 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v8 + 16);
  return *(_DWORD *)(v8 + 8) >> 8 == 0 || a3 == *(_DWORD *)(v8 + 8) >> 8;
}
