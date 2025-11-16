// Function: sub_1D91B20
// Address: 0x1d91b20
//
__int64 __fastcall sub_1D91B20(__int64 a1, char *a2, __int64 a3, char a4, unsigned int *a5, unsigned int a6)
{
  char v6; // al
  __int64 v10; // rsi
  unsigned int v11; // r14d
  __int64 v12; // rdi
  __int64 (*v13)(); // rax
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // rax

  *a5 = 0;
  v6 = *a2;
  if ( (*a2 & 2) != 0 || (v6 & 1) != 0 )
    return 0;
  v10 = *((_QWORD *)a2 + 2);
  if ( (unsigned int)((__int64)(*(_QWORD *)(v10 + 72) - *(_QWORD *)(v10 + 64)) >> 3) > 1 )
  {
    if ( (a2[1] & 1) != 0 )
      return 0;
    v11 = *((_DWORD *)a2 + 1);
    if ( (v6 & 0x10) != 0 )
    {
      if ( *((_QWORD *)a2 + 3) )
      {
        if ( !*((_DWORD *)a2 + 12) )
        {
          --v11;
          goto LABEL_11;
        }
        if ( a4 )
          goto LABEL_10;
      }
      else if ( a4 )
      {
        goto LABEL_11;
      }
      if ( *((_QWORD *)a2 + 4) )
LABEL_10:
        ++v11;
    }
LABEL_11:
    v12 = *(_QWORD *)(a1 + 544);
    v13 = *(__int64 (**)())(*(_QWORD *)v12 + 344LL);
    if ( v13 != sub_1D91890
      && ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v13)(v12, v10, v11, a6) )
    {
      *a5 = v11;
      goto LABEL_16;
    }
    return 0;
  }
LABEL_16:
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
    if ( v15 == *(_QWORD *)(v16 + 56) + 320LL || !v15 )
      return 0;
  }
  result = 1;
  if ( *(_QWORD *)(a3 + 16) != v15 )
    return 0;
  return result;
}
