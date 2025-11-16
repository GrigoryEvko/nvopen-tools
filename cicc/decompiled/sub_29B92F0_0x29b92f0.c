// Function: sub_29B92F0
// Address: 0x29b92f0
//
__int64 __fastcall sub_29B92F0(__int64 a1, char **a2)
{
  char *v2; // r9
  char *v3; // r8
  unsigned __int64 v4; // rax
  char *v5; // r14
  __int64 v6; // r13
  char *v7; // rax
  unsigned __int64 v8; // r15

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  v2 = a2[1];
  v3 = *a2;
  v4 = ((a2[3] - a2[2]) >> 3) + ((v2 - *a2) >> 3) + ((a2[5] - a2[4]) >> 3);
  if ( v4 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  v5 = 0;
  if ( v4 )
  {
    v6 = 8 * v4;
    v7 = (char *)sub_22077B0(8 * v4);
    v8 = *(_QWORD *)a1;
    v5 = v7;
    if ( (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) > 0 )
    {
      memmove(v7, *(const void **)a1, *(_QWORD *)(a1 + 8) - *(_QWORD *)a1);
    }
    else if ( !v8 )
    {
LABEL_5:
      *(_QWORD *)a1 = v5;
      *(_QWORD *)(a1 + 8) = v5;
      *(_QWORD *)(a1 + 16) = &v5[v6];
      v3 = *a2;
      v2 = a2[1];
      goto LABEL_6;
    }
    j_j___libc_free_0(v8);
    goto LABEL_5;
  }
LABEL_6:
  sub_29B7F80(a1, v5, v3, v2);
  sub_29B7F80(a1, *(char **)(a1 + 8), a2[2], a2[3]);
  sub_29B7F80(a1, *(char **)(a1 + 8), a2[4], a2[5]);
  return a1;
}
