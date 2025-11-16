// Function: sub_BCC840
// Address: 0xbcc840
//
unsigned __int64 __fastcall sub_BCC840(_QWORD *a1, const void *a2, size_t a3)
{
  _QWORD *v5; // rdi
  __int64 v6; // rax
  unsigned __int64 v7; // r12

  v5 = (_QWORD *)*a1;
  v6 = v5[330];
  v5[340] += 32LL;
  v7 = (v6 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( v5[331] >= v7 + 32 && v6 )
  {
    v5[330] = v7 + 32;
    if ( !v7 )
      goto LABEL_5;
  }
  else
  {
    v7 = sub_9D1E70((__int64)(v5 + 330), 32, 32, 4);
  }
  *(_QWORD *)v7 = a1;
  *(_QWORD *)(v7 + 8) = 15;
  *(_QWORD *)(v7 + 16) = 0;
  *(_QWORD *)(v7 + 24) = 0;
LABEL_5:
  if ( a3 )
    sub_BCB4B0((__int64 **)v7, a2, a3);
  return v7;
}
