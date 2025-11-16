// Function: sub_8C7150
// Address: 0x8c7150
//
__int64 __fastcall sub_8C7150(__int64 a1)
{
  __int64 v1; // r13
  __int64 *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 *v6; // rdx
  __int64 *v7; // rax
  __int64 v8; // rsi

  v1 = a1;
  v2 = *(__int64 **)(a1 + 32);
  if ( v2 )
    v1 = *v2;
  if ( (*(_BYTE *)(a1 + 124) & 1) == 0 )
    goto LABEL_13;
  v3 = sub_735B70(a1);
  v4 = v1;
  if ( (*(_BYTE *)(v1 + 124) & 1) == 0 )
  {
    v5 = *(__int64 **)(v3 + 32);
    if ( !v5 )
      goto LABEL_7;
    goto LABEL_6;
  }
  v4 = sub_735B70(v1);
  v5 = *(__int64 **)(v3 + 32);
  if ( v5 )
LABEL_6:
    v3 = *v5;
LABEL_7:
  v6 = *(__int64 **)(v4 + 32);
  if ( v6 )
    v4 = *v6;
  if ( v3 == v4 )
  {
LABEL_13:
    sub_8C6CA0(a1, v1, 0x1Cu, (_QWORD *)(v1 + 64));
    sub_8C6CA0(v1, a1, 0x1Cu, (_QWORD *)(a1 + 64));
    return 1;
  }
  v7 = *(__int64 **)(a1 + 32);
  v8 = a1;
  if ( v7 )
    v8 = *v7;
  sub_8C6700((__int64 *)a1, (unsigned int *)(v8 + 64), 0x42Au, 0x425u);
  return 0;
}
