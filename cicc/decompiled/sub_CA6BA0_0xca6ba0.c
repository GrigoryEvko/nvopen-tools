// Function: sub_CA6BA0
// Address: 0xca6ba0
//
void __fastcall sub_CA6BA0(unsigned int a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned int v7; // r14d
  unsigned int v8; // r13d
  char v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx

  if ( a1 <= 0xFFFF )
  {
    v11 = a2[1];
    v7 = (a1 >> 12) | 0xFFFFFFE0;
    v12 = v11 + 1;
    v9 = a1 & 0x3F | 0x80;
    v8 = (a1 >> 6) & 0x3F | 0xFFFFFF80;
    if ( (unsigned __int64)(v11 + 1) <= a2[2] )
      goto LABEL_6;
LABEL_12:
    sub_C8D290((__int64)a2, a2 + 3, v12, 1u, a5, a6);
    v11 = a2[1];
    goto LABEL_6;
  }
  if ( a1 > 0x10FFFF )
    return;
  v6 = a2[1];
  v7 = (a1 >> 12) & 0x3F | 0xFFFFFF80;
  v8 = (a1 >> 6) & 0x3F | 0xFFFFFF80;
  v9 = a1 & 0x3F | 0x80;
  if ( (unsigned __int64)(v6 + 1) > a2[2] )
  {
    sub_C8D290((__int64)a2, a2 + 3, v6 + 1, 1u, a5, a6);
    v6 = a2[1];
  }
  *(_BYTE *)(*a2 + v6) = (a1 >> 18) | 0xF0;
  v10 = a2[1];
  v11 = v10 + 1;
  v12 = v10 + 2;
  a2[1] = v11;
  if ( v12 > a2[2] )
    goto LABEL_12;
LABEL_6:
  *(_BYTE *)(*a2 + v11) = v7;
  v13 = a2[1];
  v14 = v13 + 1;
  v15 = v13 + 2;
  a2[1] = v14;
  if ( v15 > a2[2] )
  {
    sub_C8D290((__int64)a2, a2 + 3, v15, 1u, a5, a6);
    v14 = a2[1];
  }
  *(_BYTE *)(*a2 + v14) = v8;
  v16 = a2[1];
  v17 = v16 + 1;
  v18 = v16 + 2;
  a2[1] = v17;
  if ( v18 > a2[2] )
  {
    sub_C8D290((__int64)a2, a2 + 3, v18, 1u, a5, a6);
    v17 = a2[1];
  }
  *(_BYTE *)(*a2 + v17) = v9;
  ++a2[1];
}
