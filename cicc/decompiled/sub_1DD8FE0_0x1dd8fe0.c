// Function: sub_1DD8FE0
// Address: 0x1dd8fe0
//
char *__fastcall sub_1DD8FE0(__int64 a1, __int64 a2, int a3)
{
  char *v4; // r8
  _BYTE *v5; // rsi
  __int64 v6; // rdi
  int v8; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v9; // [rsp+8h] [rbp-18h] BYREF

  v9 = a2;
  v4 = *(char **)(a1 + 120);
  v8 = a3;
  if ( *(char **)(a1 + 112) != v4 || (v5 = *(_BYTE **)(a1 + 96), v5 == *(_BYTE **)(a1 + 88)) )
  {
    if ( v4 == *(char **)(a1 + 128) )
    {
      sub_1DD8E60((char **)(a1 + 112), v4, &v8);
      v5 = *(_BYTE **)(a1 + 96);
    }
    else
    {
      if ( v4 )
      {
        *(_DWORD *)v4 = v8;
        v4 = *(char **)(a1 + 120);
      }
      v5 = *(_BYTE **)(a1 + 96);
      *(_QWORD *)(a1 + 120) = v4 + 4;
    }
    if ( v5 == *(_BYTE **)(a1 + 104) )
      goto LABEL_12;
  }
  else if ( v5 == *(_BYTE **)(a1 + 104) )
  {
LABEL_12:
    sub_1D4AF10(a1 + 88, v5, &v9);
    return sub_1DD8D00(v9, (char *)a1);
  }
  v6 = v9;
  if ( v5 )
  {
    *(_QWORD *)v5 = v9;
    v5 = *(_BYTE **)(a1 + 96);
  }
  *(_QWORD *)(a1 + 96) = v5 + 8;
  return sub_1DD8D00(v6, (char *)a1);
}
