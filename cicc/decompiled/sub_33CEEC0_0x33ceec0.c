// Function: sub_33CEEC0
// Address: 0x33ceec0
//
__int64 __fastcall sub_33CEEC0(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // rdi
  __int64 v4; // rax
  unsigned __int8 (*v5)(void); // rdx
  __int64 (*v6)(); // rax
  unsigned int *v7; // rdx
  unsigned int *v8; // rsi
  __int64 v9; // rcx
  __int16 v10; // ax

  v3 = (__int64 *)a1[2];
  v4 = *v3;
  v5 = *(unsigned __int8 (**)(void))(*v3 + 1880);
  if ( (char *)v5 != (char *)sub_302E050 )
  {
    if ( v5() )
      return 0;
    v3 = (__int64 *)a1[2];
    v4 = *v3;
  }
  v6 = *(__int64 (**)())(v4 + 1856);
  if ( v6 != sub_302E040
    && ((unsigned __int8 (__fastcall *)(__int64 *, __int64, _QWORD, _QWORD))v6)(v3, a2, a1[11], a1[10]) )
  {
    return 1;
  }
  v7 = *(unsigned int **)(a2 + 40);
  v8 = &v7[10 * *(unsigned int *)(a2 + 64)];
  if ( v7 != v8 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)v7;
      v10 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16LL * v7[2]);
      if ( v10 != 1 && (*(_BYTE *)(v9 + 32) & 4) != 0 && (v10 != 262 || (unsigned int)(*(_DWORD *)(v9 + 24) - 49) > 1) )
        break;
      v7 += 10;
      if ( v8 == v7 )
        return 0;
    }
    return 1;
  }
  return 0;
}
