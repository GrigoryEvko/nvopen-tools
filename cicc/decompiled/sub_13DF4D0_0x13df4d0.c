// Function: sub_13DF4D0
// Address: 0x13df4d0
//
unsigned __int8 *__fastcall sub_13DF4D0(unsigned int a1, unsigned __int8 *a2, unsigned __int8 *a3, _QWORD *a4, int a5)
{
  unsigned int v5; // r8d
  unsigned __int8 *v8; // rbx
  unsigned __int8 *v9; // r12
  unsigned __int8 *result; // rax
  unsigned __int8 *v11; // rdx
  unsigned __int8 *v12; // rcx
  bool v13; // si
  int v14; // eax
  unsigned __int8 **v15; // rdx
  unsigned int v16; // [rsp-3Ch] [rbp-3Ch]
  unsigned int v17; // [rsp-3Ch] [rbp-3Ch]

  if ( !a5 )
    return 0;
  v5 = a5 - 1;
  v8 = a2;
  if ( a2[16] == 79 || a3 == a2 )
  {
    v17 = v5;
    v9 = (unsigned __int8 *)sub_13DDBD0(a1, *((unsigned __int8 **)a2 - 6), a3, a4, v5);
    result = (unsigned __int8 *)sub_13DDBD0(a1, *((unsigned __int8 **)a2 - 3), a3, a4, v17);
    v11 = a2;
  }
  else
  {
    v16 = v5;
    v9 = (unsigned __int8 *)sub_13DDBD0(a1, a2, *((unsigned __int8 **)a3 - 6), a4, v5);
    result = (unsigned __int8 *)sub_13DDBD0(a1, a2, *((unsigned __int8 **)a3 - 3), a4, v16);
    v11 = a3;
  }
  if ( v9 == result || v9 && v9[16] == 9 )
    return result;
  if ( !result )
  {
    v12 = (unsigned __int8 *)*((_QWORD *)v11 - 6);
    v13 = 0;
    if ( v12 != v9 )
      goto LABEL_14;
    goto LABEL_33;
  }
  if ( result[16] == 9 )
    return v9;
  v12 = (unsigned __int8 *)*((_QWORD *)v11 - 6);
  if ( v12 == v9 )
  {
LABEL_33:
    if ( *((unsigned __int8 **)v11 - 3) == result )
      return v11;
    v13 = result != 0;
    if ( v9 )
    {
LABEL_15:
      if ( !result )
        goto LABEL_16;
      return 0;
    }
    if ( result )
      goto LABEL_12;
LABEL_14:
    if ( !v9 )
      return 0;
    goto LABEL_15;
  }
  if ( v9 )
    return 0;
LABEL_12:
  v9 = result;
  v13 = 1;
LABEL_16:
  v14 = v9[16];
  if ( (unsigned __int8)v14 <= 0x17u || v14 - 24 != a1 )
    return 0;
  if ( !v13 )
    v12 = (unsigned __int8 *)*((_QWORD *)v11 - 3);
  if ( v8 == v11 )
  {
    v8 = v12;
    v12 = a3;
  }
  if ( (v9[23] & 0x40) != 0 )
    v15 = (unsigned __int8 **)*((_QWORD *)v9 - 1);
  else
    v15 = (unsigned __int8 **)&v9[-24 * (*((_DWORD *)v9 + 5) & 0xFFFFFFF)];
  if ( *v15 == v8 && v12 == v15[3] )
    return v9;
  if ( a1 <= 0x10 || (result = 0, a1 - 26 <= 1) )
  {
    if ( v15[3] != v8 || v12 != *v15 )
      return 0;
    return v9;
  }
  return result;
}
