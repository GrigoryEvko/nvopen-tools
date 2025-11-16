// Function: sub_B977E0
// Address: 0xb977e0
//
char *__fastcall sub_B977E0(char *a1, unsigned __int64 a2, int a3)
{
  char v5; // dl
  unsigned __int64 v6; // rax
  __int16 v7; // bx
  char *result; // rax
  _BYTE *v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rdx
  char *i; // rdx

  v5 = *a1;
  *((_DWORD *)a1 + 2) = 0;
  *a1 = (a3 != 0) | (2 * (a2 > 0xF)) | v5 & 0xFC;
  if ( a2 > 0xF )
  {
    *(_WORD *)a1 = *(_WORD *)a1 & 0xFC03 | 8;
    *((_QWORD *)a1 - 2) = a1;
    v10 = (__int64)(a1 - 16);
    *(_DWORD *)(v10 + 8) = 0;
    *(_DWORD *)(v10 + 12) = 0;
    sub_B97700(v10, a2);
    v11 = *((_QWORD *)a1 - 2);
    result = (char *)(v11 + 8LL * *((unsigned int *)a1 - 2));
    for ( i = (char *)(v11 + 8 * a2); i != result; result += 8 )
    {
      if ( result )
        *(_QWORD *)result = 0;
    }
    *((_DWORD *)a1 - 2) = a2;
  }
  else
  {
    v6 = 2LL * (a3 != 0);
    if ( v6 < a2 )
      v6 = a2;
    v7 = 4 * v6;
    result = (char *)(8 * v6);
    v9 = (_BYTE *)(a1 - result);
    *(_WORD *)a1 = *(_WORD *)a1 & 0xFC03 | (v7 | ((a2 & 0xF) << 6)) & 0x3FC;
    if ( a1 != (char *)(a1 - result) )
    {
      if ( (unsigned int)result < 8 )
      {
        if ( (_DWORD)result )
          *v9 = 0;
      }
      else
      {
        *(_QWORD *)v9 = 0;
        *(_QWORD *)&v9[(unsigned int)result - 8] = 0;
        memset(
          (void *)((a1 - result + 8) & 0xFFFFFFFFFFFFFFF8LL),
          0,
          8LL * (((unsigned int)a1 - (((_DWORD)a1 - (_DWORD)result + 8) & 0xFFFFFFF8)) >> 3));
        return 0;
      }
    }
  }
  return result;
}
