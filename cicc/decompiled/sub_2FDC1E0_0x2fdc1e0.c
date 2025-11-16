// Function: sub_2FDC1E0
// Address: 0x2fdc1e0
//
__int64 __fastcall sub_2FDC1E0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r9
  _DWORD *v5; // r14
  int v6; // eax
  int v7; // ecx
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // r8
  int v10; // eax
  __int64 result; // rax
  unsigned __int16 *v12; // r14
  unsigned __int16 v13; // r13
  _QWORD *v14; // rdi
  char v15; // r8
  int v16; // r15d
  char v17; // al
  __int64 v18; // r15
  __int64 (*v19)(); // rax

  v5 = (_DWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2[2] + 200LL))(a2[2]);
  v6 = v5[4];
  v7 = *(_DWORD *)(a3 + 64) & 0x3F;
  if ( v7 )
    *(_QWORD *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) - 8) &= ~(-1LL << v7);
  v8 = *(unsigned int *)(a3 + 8);
  *(_DWORD *)(a3 + 64) = v6;
  v9 = (unsigned int)(v6 + 63) >> 6;
  if ( v9 != v8 )
  {
    if ( v9 >= v8 )
    {
      v18 = v9 - v8;
      if ( v9 > *(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v9, 8u, v9, v4);
        v8 = *(unsigned int *)(a3 + 8);
      }
      if ( 8 * v18 )
      {
        memset((void *)(*(_QWORD *)a3 + 8 * v8), 0, 8 * v18);
        LODWORD(v8) = *(_DWORD *)(a3 + 8);
      }
      v6 = *(_DWORD *)(a3 + 64);
      *(_DWORD *)(a3 + 8) = v18 + v8;
    }
    else
    {
      *(_DWORD *)(a3 + 8) = (unsigned int)(v6 + 63) >> 6;
    }
  }
  v10 = v6 & 0x3F;
  if ( v10 )
    *(_QWORD *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) - 8) &= ~(-1LL << v10);
  if ( (*(_BYTE *)(a2[1] + 878) & 0x20) != 0
    && (unsigned __int8)sub_2FDC130(*a2)
    && ((v19 = *(__int64 (**)())(*(_QWORD *)a1 + 344LL), v19 == sub_2F7B4C0)
     || ((unsigned __int8 (__fastcall *)(__int64, __int64))v19)(a1, *a2)) )
  {
    result = *(_QWORD *)(*(_QWORD *)v5 + 80LL);
    if ( (__int64 (*)())result == sub_2FDBB40 )
      return result;
    result = ((__int64 (__fastcall *)(_DWORD *, __int64 *))result)(v5, a2);
    v12 = (unsigned __int16 *)result;
  }
  else
  {
    result = (__int64)sub_2EBFBC0((_QWORD *)a2[4]);
    v12 = (unsigned __int16 *)result;
  }
  if ( v12 )
  {
    if ( *v12 )
    {
      result = sub_B2D610(*a2, 20);
      if ( !(_BYTE)result )
      {
        result = sub_B2D610(*a2, 36);
        if ( !(_BYTE)result
          || (result = sub_B2D610(*a2, 41), !(_BYTE)result)
          || (result = sub_B2D610(*a2, 95), (_BYTE)result)
          || (result = *(_QWORD *)(*(_QWORD *)a1 + 88LL), (__int64 (*)())result == sub_2FDBD20)
          || (result = ((__int64 (__fastcall *)(__int64, __int64 *))result)(a1, a2), !(_BYTE)result) )
        {
          v13 = *v12;
          v14 = (_QWORD *)a2[4];
          v15 = *((_BYTE *)a2 + 577);
          if ( *v12 )
          {
            v16 = 0;
            do
            {
              if ( v15 || (v17 = sub_2EBF6F0(v14, v13, 0), v15 = 0, v17) )
                *(_QWORD *)(*(_QWORD *)a3 + 8LL * (v13 >> 6)) |= 1LL << v13;
              result = (unsigned int)(v16 + 1);
              v13 = v12[result];
              ++v16;
            }
            while ( v13 );
          }
        }
      }
    }
  }
  return result;
}
