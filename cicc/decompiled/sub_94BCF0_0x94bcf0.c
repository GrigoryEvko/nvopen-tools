// Function: sub_94BCF0
// Address: 0x94bcf0
//
_BYTE *__fastcall sub_94BCF0(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 (__fastcall *v5)(__int64, __int64, __int64); // rax
  _BYTE *v6; // r12
  unsigned int *v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned int *v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rsi
  char v16[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v17; // [rsp+20h] [rbp-30h]

  if ( a3 == *(_QWORD *)(a2 + 8) )
    return (_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    v5 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1[10] + 136LL);
    if ( v5 == sub_928970 )
      v6 = (_BYTE *)sub_ADAFB0(a2, a3);
    else
      v6 = (_BYTE *)((__int64 (__fastcall *)(unsigned int *, __int64))v5)(a1[10], a2);
    if ( *v6 > 0x1Cu )
    {
      (*(void (__fastcall **)(unsigned int *, _BYTE *, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v6,
        a4,
        a1[7],
        a1[8]);
      v7 = *a1;
      v8 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
      if ( *a1 != (unsigned int *)v8 )
      {
        do
        {
          v9 = *((_QWORD *)v7 + 1);
          v10 = *v7;
          v7 += 4;
          sub_B99FD0(v6, v10, v9);
        }
        while ( (unsigned int *)v8 != v7 );
      }
    }
    return v6;
  }
  v17 = 257;
  v6 = (_BYTE *)sub_B52210(a2, a3, v16, 0, 0);
  (*(void (__fastcall **)(unsigned int *, _BYTE *, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v6,
    a4,
    a1[7],
    a1[8]);
  v12 = *a1;
  v13 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 == (unsigned int *)v13 )
    return v6;
  do
  {
    v14 = *((_QWORD *)v12 + 1);
    v15 = *v12;
    v12 += 4;
    sub_B99FD0(v6, v15, v14);
  }
  while ( (unsigned int *)v13 != v12 );
  return v6;
}
