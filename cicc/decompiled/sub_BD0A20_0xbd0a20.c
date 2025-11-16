// Function: sub_BD0A20
// Address: 0xbd0a20
//
unsigned __int64 *__fastcall sub_BD0A20(unsigned __int64 *a1, __int64 a2, __int64 *a3, __int64 a4, char a5)
{
  unsigned int v9; // eax
  unsigned int v10; // edx
  char *v11; // rcx
  __int64 v13; // r12
  _QWORD *v14; // rdi
  __int64 v15; // rax
  _QWORD v16[7]; // [rsp+8h] [rbp-38h] BYREF

  sub_BD0200(v16, a2, a3, a4);
  if ( (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v16[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  v9 = *(_DWORD *)(a2 + 8);
  v10 = v9 >> 8;
  if ( a5 )
  {
    *(_DWORD *)(a2 + 12) = a4;
    v11 = 0;
    *(_DWORD *)(a2 + 8) = ((v10 | 3) << 8) | (unsigned __int8)v9;
    if ( !a4 )
      goto LABEL_4;
LABEL_8:
    v13 = 8 * a4;
    v14 = **(_QWORD ***)a2;
    v15 = v14[330];
    v14[340] += v13;
    v11 = (char *)((v15 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v14[331] >= (unsigned __int64)&v11[v13] && v15 )
      v14[330] = &v11[v13];
    else
      v11 = (char *)sub_9D1E70((__int64)(v14 + 330), v13, v13, 3);
    if ( v13 )
      v11 = (char *)memmove(v11, a3, v13);
    goto LABEL_4;
  }
  *(_DWORD *)(a2 + 12) = a4;
  v11 = 0;
  *(_DWORD *)(a2 + 8) = ((v10 | 1) << 8) | (unsigned __int8)v9;
  if ( a4 )
    goto LABEL_8;
LABEL_4:
  *(_QWORD *)(a2 + 16) = v11;
  *a1 = 1;
  return a1;
}
