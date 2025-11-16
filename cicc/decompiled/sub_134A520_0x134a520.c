// Function: sub_134A520
// Address: 0x134a520
//
unsigned __int64 __fastcall sub_134A520(
        __int64 a1,
        unsigned __int64 a2,
        __int64 *a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5)
{
  _QWORD *v5; // r15
  unsigned __int64 v10; // rcx
  unsigned __int64 *v11; // rax
  unsigned __int64 v12; // rdi
  _QWORD *v13; // rsi
  unsigned __int64 result; // rax
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rdx
  unsigned int i; // r8d
  _QWORD *v18; // r10
  _QWORD *v19; // r9
  _QWORD v20[54]; // [rsp+10h] [rbp-1B0h] BYREF

  v5 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    v5 = v20;
    sub_130D500(v20);
  }
  v10 = a2 & 0xFFFFFFFFC0000000LL;
  v11 = (_QWORD *)((char *)v5 + ((a2 >> 26) & 0xF0));
  v12 = *v11;
  if ( (a2 & 0xFFFFFFFFC0000000LL) == *v11 )
  {
    v13 = (_QWORD *)(v11[1] + ((a2 >> 9) & 0x1FFFF8));
  }
  else if ( v10 == v5[32] )
  {
    v5[32] = v12;
    v15 = v5[33];
    v5[33] = v11[1];
LABEL_12:
    *v11 = v10;
    v11[1] = v15;
    v13 = (_QWORD *)(v15 + ((a2 >> 9) & 0x1FFFF8));
  }
  else
  {
    v16 = v5 + 34;
    for ( i = 1; i != 8; ++i )
    {
      if ( v10 == *v16 )
      {
        v18 = &v5[2 * i - 2];
        v19 = &v5[2 * i];
        v15 = v19[33];
        v19[32] = v18[32];
        v19[33] = v18[33];
        v18[32] = v12;
        v18[33] = v11[1];
        goto LABEL_12;
      }
      v16 += 2;
    }
    v13 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v5, a2, 1, 0);
  }
  result = ((__int64)(*v13 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL;
  if ( result )
  {
    *a5 = *(_QWORD *)((((__int64)(*v13 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL) + 0x10) & 0xFFFFFFFFFFFFF000LL;
    if ( (*(_QWORD *)result & 0x1000LL) != 0 )
    {
      *a3 = (*(_QWORD *)result >> 28) & 0x3FFLL;
      result = *((unsigned int *)&unk_5260DE0 + 10 * (unsigned __int8)(*(_QWORD *)result >> 20) + 4);
      *a4 = result;
    }
    else
    {
      *a3 = 0;
      *a4 = 1;
    }
  }
  else
  {
    *a5 = 0;
    *a4 = 0;
    *a3 = 0;
  }
  return result;
}
