// Function: sub_CA6D50
// Address: 0xca6d50
//
void __fastcall sub_CA6D50(unsigned int a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx

  v6 = a1;
  if ( a1 <= 0x7F )
  {
    v9 = a2[1];
    v10 = v9 + 1;
    if ( (unsigned __int64)(v9 + 1) <= a2[2] )
      goto LABEL_7;
LABEL_9:
    sub_C8D290((__int64)a2, a2 + 3, v10, 1u, a5, a6);
    v9 = a2[1];
    goto LABEL_7;
  }
  if ( a1 > 0x7FF )
  {
    sub_CA6BA0(a1, a2, a3, a4, a5, a6);
    return;
  }
  v7 = a2[1];
  v6 = a1 & 0x3F | 0x80;
  if ( (unsigned __int64)(v7 + 1) > a2[2] )
  {
    sub_C8D290((__int64)a2, a2 + 3, v7 + 1, 1u, a5, a6);
    v7 = a2[1];
  }
  *(_BYTE *)(*a2 + v7) = (a1 >> 6) | 0xC0;
  v8 = a2[1];
  v9 = v8 + 1;
  v10 = v8 + 2;
  a2[1] = v9;
  if ( v10 > a2[2] )
    goto LABEL_9;
LABEL_7:
  *(_BYTE *)(*a2 + v9) = v6;
  ++a2[1];
}
