// Function: sub_1B420C0
// Address: 0x1b420c0
//
unsigned __int64 __fastcall sub_1B420C0(__int64 *a1, __int64 *a2)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rsi
  unsigned __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax

  v3 = sub_1648700((__int64)a2);
  if ( *((_BYTE *)v3 + 16) == 77 )
  {
    if ( (*((_BYTE *)v3 + 23) & 0x40) != 0 )
      v4 = (_QWORD *)*(v3 - 1);
    else
      v4 = &v3[-3 * (*((_DWORD *)v3 + 5) & 0xFFFFFFF)];
    result = sub_1B3FB80(a1, v4[3 * *((unsigned int *)v3 + 14) + 1 + -1431655765 * (unsigned int)(a2 - v4)]);
    v6 = result;
  }
  else
  {
    result = sub_1B40B40(a1, v3[5]);
    v6 = result;
  }
  v7 = *a2;
  if ( v6 == *a2 )
  {
LABEL_16:
    if ( !v7 )
      goto LABEL_9;
    goto LABEL_7;
  }
  if ( (*(_BYTE *)(v7 + 17) & 1) != 0 )
  {
    result = sub_164CC90(v7, v6);
    v7 = *a2;
    goto LABEL_16;
  }
LABEL_7:
  v8 = a2[1];
  result = a2[2] & 0xFFFFFFFFFFFFFFFCLL;
  *(_QWORD *)result = v8;
  if ( v8 )
  {
    result |= *(_QWORD *)(v8 + 16) & 3LL;
    *(_QWORD *)(v8 + 16) = result;
  }
LABEL_9:
  *a2 = v6;
  if ( v6 )
  {
    v9 = *(_QWORD *)(v6 + 8);
    a2[1] = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = (unsigned __int64)(a2 + 1) | *(_QWORD *)(v9 + 16) & 3LL;
    result = (v6 + 8) | a2[2] & 3;
    a2[2] = result;
    *(_QWORD *)(v6 + 8) = a2;
  }
  return result;
}
