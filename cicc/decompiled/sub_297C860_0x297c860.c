// Function: sub_297C860
// Address: 0x297c860
//
bool __fastcall sub_297C860(int *a1)
{
  int v1; // edx
  bool result; // al
  __int64 v4; // rdi
  unsigned int v5; // ebx
  __int64 v6; // rdi
  unsigned int v7; // ebx
  int v8; // r8d
  __int64 v9; // rdi
  unsigned int v10; // r13d
  __int64 v11; // r13
  _QWORD *v12; // rbx
  unsigned int v13; // r12d
  _BYTE *v14; // rdi
  unsigned int v15; // r14d
  __int64 v16; // rdx
  __int64 v17; // rax

  v1 = *a1;
  if ( *a1 == 1 )
  {
    v6 = *((_QWORD *)a1 + 2);
    v7 = *(_DWORD *)(v6 + 32);
    if ( v7 <= 0x40 )
    {
      v16 = *(_QWORD *)(v6 + 24);
      result = 1;
      if ( v16 != 1 && v7 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) == v16;
    }
    else
    {
      v8 = sub_C444A0(v6 + 24);
      result = 1;
      if ( v8 != v7 - 1 )
        return v7 == (unsigned int)sub_C445E0(v6 + 24);
    }
  }
  else if ( v1 == 2 )
  {
    v4 = *((_QWORD *)a1 + 2);
    v5 = *(_DWORD *)(v4 + 32);
    if ( v5 <= 0x40 )
      return *(_QWORD *)(v4 + 24) == 0;
    else
      return v5 == (unsigned int)sub_C444A0(v4 + 24);
  }
  else
  {
    result = 0;
    if ( v1 != 3 )
      return result;
    v9 = *((_QWORD *)a1 + 2);
    v10 = *(_DWORD *)(v9 + 32);
    if ( v10 <= 0x40 )
    {
      v17 = *(_QWORD *)(v9 + 24);
      if ( v17 == 1 || !v10 )
        goto LABEL_13;
      result = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) == v17;
    }
    else
    {
      if ( v10 - 1 == (unsigned int)sub_C444A0(v9 + 24) )
      {
LABEL_13:
        v11 = *((_QWORD *)a1 + 4);
        v12 = (_QWORD *)(v11 + 32 * (1LL - (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)));
        if ( (_QWORD *)v11 == v12 )
          return 1;
        v13 = 0;
        while ( 1 )
        {
          v14 = (_BYTE *)*v12;
          if ( *(_BYTE *)*v12 != 17 )
            goto LABEL_16;
          v15 = *((_DWORD *)v14 + 8);
          if ( v15 <= 0x40 )
            break;
          if ( v15 != (unsigned int)sub_C444A0((__int64)(v14 + 24)) )
            goto LABEL_16;
LABEL_17:
          v12 += 4;
          if ( v12 == (_QWORD *)v11 )
            return v13 <= 1;
        }
        if ( !*((_QWORD *)v14 + 3) )
          goto LABEL_17;
LABEL_16:
        ++v13;
        goto LABEL_17;
      }
      result = v10 == (unsigned int)sub_C445E0(v9 + 24);
    }
    if ( result )
      goto LABEL_13;
  }
  return result;
}
