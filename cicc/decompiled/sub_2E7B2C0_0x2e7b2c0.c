// Function: sub_2E7B2C0
// Address: 0x2e7b2c0
//
_QWORD *__fastcall sub_2E7B2C0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r13
  __int64 v4; // rcx
  unsigned __int64 v5; // rax

  v2 = (_QWORD *)a1[28];
  if ( v2 )
  {
    a1[28] = *v2;
LABEL_3:
    sub_2E90E00(v2, a1, a2);
    return v2;
  }
  v4 = a1[16];
  a1[26] += 72LL;
  v5 = (v4 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[17] < v5 + 72 || !v4 )
  {
    v5 = sub_9D1E70((__int64)(a1 + 16), 72, 72, 3);
    goto LABEL_9;
  }
  a1[16] = v5 + 72;
  if ( v5 )
  {
LABEL_9:
    v2 = (_QWORD *)v5;
    goto LABEL_3;
  }
  return 0;
}
