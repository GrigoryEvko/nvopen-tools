// Function: sub_271D550
// Address: 0x271d550
//
void *__fastcall sub_271D550(unsigned __int8 *a1, _BYTE *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  int v8; // edi
  int v9; // eax
  __int64 v10; // rdx
  void *result; // rax

  v6 = a3;
  v8 = (unsigned __int8)a2[2];
  v9 = a1[2];
  v10 = *a1;
  if ( v9 == v8 )
  {
    a1[2] = v8;
    LOBYTE(v10) = *a2 & v10;
    *a1 = v10;
    if ( !v9 )
      goto LABEL_11;
    goto LABEL_13;
  }
  if ( !a1[2] || !a2[2] )
    goto LABEL_10;
  a6 = a1[2];
  if ( (unsigned __int8)v8 < (unsigned __int8)v9 )
  {
    v9 = (unsigned __int8)a2[2];
    v8 = a1[2];
  }
  if ( (_BYTE)v6 )
  {
    if ( (unsigned int)(v9 - 1) > 1 || (unsigned int)(v8 - 2) > 1 )
      goto LABEL_10;
    LOBYTE(v9) = v8;
LABEL_19:
    a1[2] = v9;
    LOBYTE(v10) = *a2 & v10;
    *a1 = v10;
    goto LABEL_13;
  }
  v6 = (unsigned int)(v9 - 2);
  if ( (unsigned int)v6 <= 1 )
  {
    if ( (unsigned int)(v8 - 3) > 2 )
    {
LABEL_10:
      a1[2] = 0;
      *a1 = *a2 & v10;
LABEL_11:
      a1[1] = 0;
      return sub_271CF50((__int64)(a1 + 8), (__int64)a2);
    }
    goto LABEL_19;
  }
  if ( v8 != 5 || v9 != 4 )
    goto LABEL_10;
  a1[2] = 4;
  LOBYTE(v10) = *a2 & v10;
  *a1 = v10;
LABEL_13:
  if ( a1[1] || a2[1] )
    return sub_271D520((__int64)a1, 0);
  result = (void *)sub_271D020((__int64)(a1 + 8), (__int64)(a2 + 8), v10, v6, a5, a6);
  a1[1] = (unsigned __int8)result;
  return result;
}
