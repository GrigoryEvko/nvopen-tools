// Function: sub_7E1B70
// Address: 0x7e1b70
//
__int64 __fastcall sub_7E1B70(char *src, __int64 a2, __int64 a3, __int64 *a4)
{
  size_t v6; // rax
  char *v7; // rax
  char *v8; // r13
  _QWORD *v9; // rax
  _QWORD *v10; // rbx
  __int64 result; // rax

  v6 = strlen(src);
  v7 = (char *)sub_7E1510(v6 + 1);
  v8 = strcpy(v7, src);
  v9 = sub_725D60();
  *((_BYTE *)v9 + 144) |= 0x40u;
  v10 = v9;
  v9[1] = v8;
  v9[15] = a2;
  sub_877E20(0, v9, a3);
  result = *a4;
  if ( *a4 )
    *(_QWORD *)(result + 112) = v10;
  else
    *(_QWORD *)(a3 + 160) = v10;
  *a4 = (__int64)v10;
  return result;
}
