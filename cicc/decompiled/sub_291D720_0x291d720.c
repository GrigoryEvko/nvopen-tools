// Function: sub_291D720
// Address: 0x291d720
//
_BYTE *__fastcall sub_291D720(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 (__fastcall *v5)(__int64, unsigned __int64, __int64); // rax
  _BYTE *v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rdx
  unsigned int v10; // esi
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // esi
  _BYTE v16[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v17; // [rsp+20h] [rbp-30h]

  if ( a3 == *(_QWORD *)(a2 + 8) )
    return (_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    v5 = *(__int64 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)a1[10] + 144LL);
    if ( v5 == sub_B32D70 )
      v6 = (_BYTE *)sub_ADB060(a2, a3);
    else
      v6 = (_BYTE *)((__int64 (__fastcall *)(__int64, unsigned __int64))v5)(a1[10], a2);
    if ( *v6 > 0x1Cu )
    {
      (*(void (__fastcall **)(__int64, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v6,
        a4,
        a1[7],
        a1[8]);
      v7 = *a1;
      v8 = *a1 + 16LL * *((unsigned int *)a1 + 2);
      if ( *a1 != v8 )
      {
        do
        {
          v9 = *(_QWORD *)(v7 + 8);
          v10 = *(_DWORD *)v7;
          v7 += 16;
          sub_B99FD0((__int64)v6, v10, v9);
        }
        while ( v8 != v7 );
      }
    }
    return v6;
  }
  v17 = 257;
  v6 = (_BYTE *)sub_B52190(a2, a3, (__int64)v16, 0, 0);
  (*(void (__fastcall **)(__int64, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v6,
    a4,
    a1[7],
    a1[8]);
  v12 = *a1;
  v13 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 == v13 )
    return v6;
  do
  {
    v14 = *(_QWORD *)(v12 + 8);
    v15 = *(_DWORD *)v12;
    v12 += 16;
    sub_B99FD0((__int64)v6, v15, v14);
  }
  while ( v13 != v12 );
  return v6;
}
