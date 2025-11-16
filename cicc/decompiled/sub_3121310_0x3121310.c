// Function: sub_3121310
// Address: 0x3121310
//
__int64 __fastcall sub_3121310(__int64 **a1, __int64 *a2, __int64 *a3, char a4)
{
  char v6; // r12
  char v7; // al
  unsigned __int64 v8; // rdx
  int v9; // ecx
  unsigned int v10; // eax
  __int64 result; // rax
  int v12; // edx
  int v13; // r8d

  v6 = sub_A73170(a3, 54);
  v7 = sub_A73170(a3, 79);
  if ( !v6 && !v7 )
  {
    result = sub_A7A4C0(a2, a1[1], *a3);
    *a2 = result;
    return result;
  }
  v8 = *((unsigned int *)*a1 + 220);
  v9 = v8 - 24;
  v10 = (v8 - 31) & 0xFFFFFFFD;
  LOBYTE(v10) = v10 == 0;
  LOBYTE(v9) = (unsigned int)(v8 - 24) <= 1;
  result = v9 | v10;
  if ( !a4 )
  {
    if ( (_BYTE)result )
    {
      v13 = v6 == 0 ? 79 : 54;
    }
    else
    {
      result = (unsigned int)(v8 - 13);
      v13 = 54;
      if ( (unsigned int)result > 1 && (_DWORD)v8 != 29 )
        return result;
    }
    result = sub_A7A580(a2, a1[1], v13);
    *a2 = result;
    return result;
  }
  if ( (_BYTE)result )
  {
    v12 = v6 == 0 ? 79 : 54;
LABEL_6:
    result = sub_A7A580(a2, a1[1], v12);
    *a2 = result;
    return result;
  }
  if ( (unsigned int)v8 <= 0x1D )
  {
    result = 537878528;
    if ( _bittest64(&result, v8) )
    {
      v12 = 54;
      goto LABEL_6;
    }
  }
  return result;
}
