// Function: sub_305F2F0
// Address: 0x305f2f0
//
__int64 __fastcall sub_305F2F0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  int v3; // eax
  unsigned int v5; // r8d
  unsigned __int8 *v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 (*v11)(); // r9

  v3 = *a2;
  if ( (unsigned int)(v3 - 29) > 0x14 )
  {
    v5 = 0;
    if ( (unsigned int)(v3 - 51) > 1 )
      return v5;
  }
  else
  {
    v5 = 0;
    if ( (unsigned int)(v3 - 29) <= 0x12 )
      return v5;
  }
  if ( (a2[7] & 0x40) != 0 )
    v7 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v7 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v5 = 0;
  if ( **((_BYTE **)v7 + 4) == 17 )
  {
    v8 = sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), *((__int64 **)a2 + 1), 0);
    v10 = *(_QWORD *)(a1 + 32);
    v5 = 1;
    v11 = *(__int64 (**)())(*(_QWORD *)v10 + 200LL);
    if ( v11 != sub_2FE2F30 )
      return ((unsigned int (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v11)(
               v10,
               v8,
               v9,
               *(_QWORD *)(a3 + 120),
               1)
           ^ 1;
  }
  return v5;
}
