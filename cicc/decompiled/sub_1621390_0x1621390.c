// Function: sub_1621390
// Address: 0x1621390
//
char *__fastcall sub_1621390(char *a1)
{
  char v1; // al
  char *v2; // rax
  char *result; // rax
  _BYTE *v4; // rsi
  char *v5; // [rsp+8h] [rbp-8h] BYREF

  v1 = *a1;
  a1[1] = 1;
  if ( v1 == 8 || v1 == 4 )
    *((_DWORD *)a1 + 1) = 0;
  v2 = (char *)(*((_QWORD *)a1 + 2) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*((_QWORD *)a1 + 2) & 4) != 0 )
    v2 = *(char **)v2;
  result = *(char **)v2;
  v5 = a1;
  v4 = (_BYTE *)*((_QWORD *)result + 188);
  if ( v4 == *((_BYTE **)result + 189) )
    return sub_1621200((__int64)(result + 1496), v4, &v5);
  if ( v4 )
  {
    *(_QWORD *)v4 = a1;
    v4 = (_BYTE *)*((_QWORD *)result + 188);
  }
  *((_QWORD *)result + 188) = v4 + 8;
  return result;
}
