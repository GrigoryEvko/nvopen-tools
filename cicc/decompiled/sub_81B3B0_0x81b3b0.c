// Function: sub_81B3B0
// Address: 0x81b3b0
//
unsigned __int8 *__fastcall sub_81B3B0(void *src, size_t n, unsigned __int8 **a3)
{
  _DWORD *v3; // rcx
  __int64 v4; // rax
  unsigned __int8 *v6; // rdx
  unsigned __int64 v7; // rbx
  size_t v8; // rdx
  int v9; // ebx
  char *v10; // rcx
  unsigned __int8 *result; // rax

  v3 = qword_4F195A0;
  v4 = qword_4F195A8;
  v6 = *a3;
  if ( v6 )
  {
    v7 = v6[1] | ((unsigned __int64)v6[3] << 16) | ((unsigned __int64)v6[2] << 8);
    v8 = 0xFFFFFF - v7;
    v9 = n + v7;
    if ( v8 >= n )
    {
      if ( qword_4F195A8 - (__int64)qword_4F195A0 >= n )
        goto LABEL_4;
LABEL_8:
      sub_81AC10(n);
      v3 = qword_4F195A0;
      goto LABEL_4;
    }
  }
  *a3 = (unsigned __int8 *)qword_4F195A0;
  if ( (unsigned __int64)(v4 - (_QWORD)v3) <= 3 )
  {
    sub_81AC10(4u);
    v3 = qword_4F195A0;
    v4 = qword_4F195A8;
  }
  *v3 = 1;
  v9 = n;
  qword_4F195A0 = ++v3;
  if ( v4 - (__int64)v3 < n )
    goto LABEL_8;
LABEL_4:
  v10 = (char *)memcpy(v3, src, n);
  result = *a3;
  *(_WORD *)(result + 1) = v9;
  result[3] = BYTE2(v9);
  qword_4F195A0 = &v10[n];
  return result;
}
