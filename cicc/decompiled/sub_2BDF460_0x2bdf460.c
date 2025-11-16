// Function: sub_2BDF460
// Address: 0x2bdf460
//
unsigned __int64 __fastcall sub_2BDF460(unsigned __int8 *a1)
{
  unsigned __int8 *v2; // rax
  __int64 v3; // r14
  unsigned int v4; // r13d
  __int64 v5; // r15
  unsigned __int8 v6; // r12
  int v7; // esi
  __int64 (__fastcall *v8)(__int64, unsigned int); // rcx
  signed __int8 v9; // al
  unsigned __int64 result; // rax
  char v11; // r8
  __int64 v12; // r14
  __int64 v13; // r15
  unsigned __int8 *v14; // r13
  __int64 (__fastcall *v15)(__int64, unsigned int); // rax
  int v16; // eax
  _BYTE *v17; // rdx
  unsigned __int64 *v18; // rdi
  __int64 v19; // rdx
  __int64 (__fastcall *v20)(unsigned __int8 *); // rax
  unsigned __int8 *v21; // rdi

  v2 = (unsigned __int8 *)*((_QWORD *)a1 + 22);
  v3 = *((_QWORD *)a1 + 24);
  *((_QWORD *)a1 + 22) = v2 + 1;
  v4 = (char)*v2;
  v5 = *v2;
  v6 = *v2;
  v7 = *(char *)(v3 + v5 + 313);
  if ( !*(_BYTE *)(v3 + v5 + 313) )
  {
    v7 = (char)*v2;
    v8 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v3 + 64LL);
    v9 = *v2;
    if ( v8 != sub_2216C50 )
    {
      v9 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v8)(v3, v4, 32);
      v7 = v9;
    }
    if ( v9 != 32 )
      *(_BYTE *)(v3 + v5 + 313) = v9;
  }
  result = (unsigned __int64)strchr(*((const char **)a1 + 20), v7);
  if ( !result )
  {
    *((_DWORD *)a1 + 36) = 1;
    v19 = *((_QWORD *)a1 + 26);
    v18 = (unsigned __int64 *)(a1 + 200);
    v11 = v4;
    return (unsigned __int64)sub_2240FD0(v18, 0, v19, 1u, v11);
  }
  if ( v6 == 92 )
  {
    result = *((_QWORD *)a1 + 22);
    if ( result == *((_QWORD *)a1 + 23) )
      goto LABEL_52;
    if ( (*((_DWORD *)a1 + 35) & 0x120) == 0 || (unsigned __int8)(*(_BYTE *)result - 40) > 1u && *(_BYTE *)result != 123 )
    {
      v20 = (__int64 (__fastcall *)(unsigned __int8 *))*((_QWORD *)a1 + 29);
      v21 = &a1[*((_QWORD *)a1 + 30)];
      if ( ((unsigned __int8)v20 & 1) != 0 )
        v20 = *(__int64 (__fastcall **)(unsigned __int8 *))((char *)v20 + *(_QWORD *)v21 - 1);
      return v20(v21);
    }
    *((_QWORD *)a1 + 22) = result + 1;
    v6 = *(_BYTE *)result;
  }
  switch ( v6 )
  {
    case '(':
      v16 = *((_DWORD *)a1 + 35);
      if ( (v16 & 0x10) == 0 || (v17 = (_BYTE *)*((_QWORD *)a1 + 22), *v17 != 63) )
      {
        result = (unsigned int)((v16 & 2) != 0) + 5;
        *((_DWORD *)a1 + 36) = result;
        return result;
      }
      *((_QWORD *)a1 + 22) = v17 + 1;
      if ( v17 + 1 != *((_BYTE **)a1 + 23) )
      {
        result = (unsigned __int8)v17[1];
        switch ( (_BYTE)result )
        {
          case ':':
            *((_DWORD *)a1 + 36) = 6;
            *((_QWORD *)a1 + 22) = v17 + 2;
            return result;
          case '=':
            v18 = (unsigned __int64 *)(a1 + 200);
            *((_DWORD *)a1 + 36) = 7;
            v11 = 112;
            *((_QWORD *)a1 + 22) = v17 + 2;
            v19 = *((_QWORD *)a1 + 26);
            return (unsigned __int64)sub_2240FD0(v18, 0, v19, 1u, v11);
          case '!':
            v18 = (unsigned __int64 *)(a1 + 200);
            *((_DWORD *)a1 + 36) = 7;
            v11 = 110;
            *((_QWORD *)a1 + 22) = v17 + 2;
            v19 = *((_QWORD *)a1 + 26);
            return (unsigned __int64)sub_2240FD0(v18, 0, v19, 1u, v11);
        }
      }
LABEL_52:
      abort();
    case ')':
      *((_DWORD *)a1 + 36) = 8;
      return result;
    case '[':
      a1[168] = 1;
      result = *((_QWORD *)a1 + 22);
      *((_DWORD *)a1 + 34) = 2;
      if ( result != *((_QWORD *)a1 + 23) && *(_BYTE *)result == 94 )
      {
        *((_DWORD *)a1 + 36) = 10;
        *((_QWORD *)a1 + 22) = ++result;
      }
      else
      {
        *((_DWORD *)a1 + 36) = 9;
      }
      return result;
    case '{':
      *((_DWORD *)a1 + 34) = 1;
      *((_DWORD *)a1 + 36) = 12;
      return result;
  }
  v11 = v6;
  if ( (v6 & 0xDF) == 0x5D )
  {
    v19 = *((_QWORD *)a1 + 26);
    v18 = (unsigned __int64 *)(a1 + 200);
    *((_DWORD *)a1 + 36) = 1;
    return (unsigned __int64)sub_2240FD0(v18, 0, v19, 1u, v11);
  }
  v12 = *((_QWORD *)a1 + 24);
  v13 = v6;
  v14 = a1;
  if ( *(_BYTE *)(v12 + v6 + 313) )
  {
    v6 = *(_BYTE *)(v12 + v6 + 313);
  }
  else
  {
    v15 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v12 + 64LL);
    if ( v15 != sub_2216C50 )
      v6 = ((__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))v15)(*((_QWORD *)a1 + 24), (unsigned int)(char)v6, 0);
    if ( v6 )
      *(_BYTE *)(v12 + v13 + 313) = v6;
  }
  result = *a1;
  if ( (_BYTE)result )
  {
    while ( v6 != (_BYTE)result )
    {
      result = v14[8];
      v14 += 8;
      if ( !(_BYTE)result )
        return result;
    }
    result = *((unsigned int *)v14 + 1);
    *((_DWORD *)a1 + 36) = result;
  }
  return result;
}
