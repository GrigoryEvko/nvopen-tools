// Function: sub_109E080
// Address: 0x109e080
//
__int64 __fastcall sub_109E080(__int64 a1, char *a2)
{
  char v3; // r13
  __int64 v4; // r15
  void *v5; // rax
  void *v6; // r14
  __int64 v7; // rdi
  __int64 result; // rax
  char v9; // dl
  __int64 *v10; // rsi
  __int64 v11; // rdi
  _QWORD *i; // rbx
  unsigned __int8 *v13; // rdi
  void *v14; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v15; // [rsp+18h] [rbp-48h]

  v3 = *a2;
  if ( *a2 )
  {
    if ( !*(_BYTE *)a1 )
    {
      v4 = *((_QWORD *)a2 + 1);
      sub_109CFD0(a1, v4);
      v3 = *a2;
      goto LABEL_4;
    }
  }
  else
  {
    result = (unsigned int)*((__int16 *)a2 + 1);
    if ( *((_WORD *)a2 + 1) == 1 )
      return result;
    v9 = *(_BYTE *)a1;
    if ( *((_WORD *)a2 + 1) == 0xFFFF )
    {
      if ( v9 )
      {
        v13 = (unsigned __int8 *)(a1 + 8);
        if ( *(void **)(a1 + 8) == sub_C33340() )
          return sub_C3CCB0((__int64)v13);
        else
          return sub_C34440(v13);
      }
      else
      {
        *(_WORD *)(a1 + 2) = -*(_WORD *)(a1 + 2);
      }
      return result;
    }
    if ( !v9 )
    {
      result = (unsigned int)(*(__int16 *)(a1 + 2) * (_DWORD)result);
      *(_WORD *)(a1 + 2) = result;
      return result;
    }
  }
  v4 = *(_QWORD *)(a1 + 8);
LABEL_4:
  v5 = sub_C33340();
  v6 = v5;
  if ( v3 )
  {
    v10 = (__int64 *)(a2 + 8);
    v11 = a1 + 8;
    if ( *(void **)(a1 + 8) == v5 )
      return sub_C3F5C0(v11, v10, 1u);
    else
      return sub_C3B950(v11, (__int64)v10, 1);
  }
  else
  {
    sub_109DF40(&v14, v4, *((__int16 *)a2 + 1));
    v7 = a1 + 8;
    if ( *(void **)(a1 + 8) == v6 )
      sub_C3F5C0(v7, (__int64 *)&v14, 1u);
    else
      sub_C3B950(v7, (__int64)&v14, 1);
    if ( v14 == v6 )
    {
      result = (__int64)v15;
      if ( v15 )
      {
        for ( i = &v15[3 * *(v15 - 1)]; v15 != i; sub_91D830(i) )
          i -= 3;
        return j_j_j___libc_free_0_0(i - 1);
      }
    }
    else
    {
      return sub_C338F0((__int64)&v14);
    }
  }
  return result;
}
