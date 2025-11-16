// Function: sub_3749A00
// Address: 0x3749a00
//
__int64 __fastcall sub_3749A00(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r15d
  __int64 v4; // rdi
  __int64 v5; // rdx
  unsigned int v6; // r14d
  unsigned int v7; // ebx
  __int64 v9; // rdx
  __int64 *v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // r8
  __int64 (*v13)(); // rax
  unsigned int v14; // edx

  v4 = a1[16];
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(_QWORD *)(a2 - 8);
  else
    v5 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v6 = sub_2D5BAE0(v4, a1[14], *(__int64 **)(*(_QWORD *)v5 + 8LL), 0);
  v7 = sub_2D5BAE0(a1[16], a1[14], *(__int64 **)(a2 + 8), 0);
  if ( (_WORD)v7 == 1 )
    return 0;
  if ( (unsigned __int16)v6 <= 1u )
    return 0;
  v9 = a1[16];
  LOBYTE(v2) = (_WORD)v7 != 0 && *(_QWORD *)(v9 + 8LL * (unsigned __int16)v6 + 112) != 0;
  if ( !(_BYTE)v2 || !*(_QWORD *)(v9 + 8LL * (unsigned __int16)v7 + 112) )
    return 0;
  v10 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
      ? *(__int64 **)(a2 - 8)
      : (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v11 = sub_3746830(a1, *v10);
  v12 = v11;
  if ( !v11 )
    return 0;
  if ( (_WORD)v6 == (_WORD)v7 )
  {
    sub_3742B00((__int64)a1, (_BYTE *)a2, v11, 1);
    return v2;
  }
  v13 = *(__int64 (**)())(*a1 + 64);
  if ( v13 != sub_3740EE0
    && (v14 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, __int64))v13)(a1, v6, v7, 234, v12)) != 0 )
  {
    sub_3742B00((__int64)a1, (_BYTE *)a2, v14, 1);
  }
  else
  {
    return 0;
  }
  return v2;
}
