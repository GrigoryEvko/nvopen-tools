// Function: sub_B95A20
// Address: 0xb95a20
//
char *__fastcall sub_B95A20(unsigned __int8 *a1)
{
  __int64 v1; // rbp
  unsigned __int8 v2; // al
  char *v3; // rax
  char *result; // rax
  _BYTE *v5; // rsi
  _QWORD v6[2]; // [rsp-10h] [rbp-10h] BYREF

  a1[1] = a1[1] & 0x80 | 1;
  v2 = *a1;
  if ( *a1 == 9 )
  {
LABEL_14:
    *((_DWORD *)a1 + 1) = 0;
    goto LABEL_5;
  }
  if ( v2 <= 9u )
  {
    if ( v2 != 5 )
    {
      if ( (unsigned __int8)(v2 - 6) <= 2u )
        goto LABEL_5;
LABEL_15:
      BUG();
    }
    goto LABEL_14;
  }
  if ( (unsigned __int8)(v2 - 10) > 0x1Au )
    goto LABEL_15;
LABEL_5:
  v6[1] = v1;
  v3 = (char *)(*((_QWORD *)a1 + 1) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*((_QWORD *)a1 + 1) & 4) != 0 )
    v3 = *(char **)v3;
  result = *(char **)v3;
  v6[0] = a1;
  v5 = (_BYTE *)*((_QWORD *)result + 209);
  if ( v5 == *((_BYTE **)result + 210) )
    return sub_B95890((__int64)(result + 1664), v5, v6);
  if ( v5 )
  {
    *(_QWORD *)v5 = a1;
    v5 = (_BYTE *)*((_QWORD *)result + 209);
  }
  *((_QWORD *)result + 209) = v5 + 8;
  return result;
}
