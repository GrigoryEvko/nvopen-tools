// Function: sub_34E6540
// Address: 0x34e6540
//
__int64 __fastcall sub_34E6540(__int64 a1, char *a2, __int64 a3, char a4, unsigned int *a5, unsigned int a6)
{
  __int64 v7; // rsi
  char v8; // al
  unsigned int v11; // r14d
  __int64 v12; // rdi
  __int64 (*v13)(); // rax
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // rax

  *a5 = 0;
  v7 = *((_QWORD *)a2 + 2);
  if ( v7 == *(_QWORD *)(a3 + 16) )
    return 0;
  v8 = *a2;
  if ( (*a2 & 2) != 0 || (v8 & 1) != 0 )
    return 0;
  if ( *(_DWORD *)(v7 + 72) <= 1u )
    goto LABEL_14;
  if ( (a2[1] & 1) != 0 )
    return 0;
  v11 = *((_DWORD *)a2 + 1);
  if ( (v8 & 0x10) != 0 )
  {
    if ( *((_QWORD *)a2 + 3) )
    {
      if ( !*((_DWORD *)a2 + 12) )
      {
        --v11;
        goto LABEL_10;
      }
      if ( a4 )
      {
LABEL_26:
        ++v11;
        goto LABEL_10;
      }
    }
    else if ( a4 )
    {
      goto LABEL_10;
    }
    if ( !*((_QWORD *)a2 + 4) )
      goto LABEL_10;
    goto LABEL_26;
  }
LABEL_10:
  v12 = *(_QWORD *)(a1 + 528);
  v13 = *(__int64 (**)())(*(_QWORD *)v12 + 432LL);
  if ( v13 == sub_2FDC550 || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v13)(v12, v7, v11, a6) )
    return 0;
  *a5 = v11;
LABEL_14:
  v15 = *((_QWORD *)a2 + 3);
  if ( a4 )
    v15 = *((_QWORD *)a2 + 4);
  if ( !v15 )
  {
    if ( (*a2 & 0x10) == 0 )
      return 0;
    if ( *((_QWORD *)a2 + 3) )
      return 0;
    v16 = *((_QWORD *)a2 + 2);
    v15 = *(_QWORD *)(v16 + 8);
    if ( v15 == *(_QWORD *)(v16 + 32) + 320LL || !v15 )
      return 0;
  }
  result = 1;
  if ( *(_QWORD *)(a3 + 16) != v15 )
    return 0;
  return result;
}
