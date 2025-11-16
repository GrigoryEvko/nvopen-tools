// Function: sub_2F8FF90
// Address: 0x2f8ff90
//
void __fastcall sub_2F8FF90(__int64 a1, unsigned __int64 a2)
{
  const void *v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  char *v5; // r15
  signed __int64 v6; // rdx

  if ( a2 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  v2 = *(const void **)a1;
  if ( a2 > (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)a1) >> 3 )
  {
    v3 = *(_QWORD *)(a1 + 8) - (_QWORD)v2;
    if ( a2 )
    {
      v4 = sub_22077B0(8 * a2);
      v2 = *(const void **)a1;
      v5 = (char *)v4;
      v6 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
      if ( v6 <= 0 )
        goto LABEL_5;
    }
    else
    {
      v6 = *(_QWORD *)(a1 + 8) - (_QWORD)v2;
      v5 = 0;
      if ( v3 <= 0 )
      {
LABEL_5:
        if ( !v2 )
        {
LABEL_6:
          *(_QWORD *)a1 = v5;
          *(_QWORD *)(a1 + 8) = &v5[v3];
          *(_QWORD *)(a1 + 16) = &v5[8 * a2];
          return;
        }
LABEL_10:
        j_j___libc_free_0((unsigned __int64)v2);
        goto LABEL_6;
      }
    }
    memmove(v5, v2, v6);
    goto LABEL_10;
  }
}
