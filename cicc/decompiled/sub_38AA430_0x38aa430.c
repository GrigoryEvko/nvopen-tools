// Function: sub_38AA430
// Address: 0x38aa430
//
__int64 __fastcall sub_38AA430(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r15d
  int v11; // r8d
  int v12; // r9d
  __int64 v14; // r15
  int v15; // eax
  unsigned __int64 v16; // rsi
  __int64 v17; // rdx
  int v18; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 v19[2]; // [rsp+20h] [rbp-50h] BYREF
  char v20; // [rsp+30h] [rbp-40h]
  char v21; // [rsp+31h] [rbp-3Fh]

  if ( *(_DWORD *)(a1 + 64) == 376 )
  {
    while ( 1 )
    {
      v10 = sub_38AA270(a1, &v18, v19, a3, a4, a5, a6, a7, a8, a9, a10);
      if ( (_BYTE)v10 )
        break;
      sub_1625C10(a2, v18, v19[0]);
      if ( v18 == 1 )
      {
        v17 = *(unsigned int *)(a1 + 208);
        if ( (unsigned int)v17 >= *(_DWORD *)(a1 + 212) )
        {
          sub_16CD150(a1 + 200, (const void *)(a1 + 216), 0, 8, v11, v12);
          v17 = *(unsigned int *)(a1 + 208);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * v17) = a2;
        ++*(_DWORD *)(a1 + 208);
      }
      if ( *(_DWORD *)(a1 + 64) != 4 )
        break;
      v14 = a1 + 8;
      v15 = sub_3887100(a1 + 8);
      *(_DWORD *)(a1 + 64) = v15;
      if ( v15 != 376 )
        goto LABEL_7;
    }
  }
  else
  {
    v14 = a1 + 8;
LABEL_7:
    v16 = *(_QWORD *)(a1 + 56);
    v21 = 1;
    v20 = 3;
    v19[0] = (__int64)"expected metadata after comma";
    return (unsigned int)sub_38814C0(v14, v16, (__int64)v19);
  }
  return v10;
}
