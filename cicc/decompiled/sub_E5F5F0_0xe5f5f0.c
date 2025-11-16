// Function: sub_E5F5F0
// Address: 0xe5f5f0
//
void __fastcall sub_E5F5F0(unsigned int a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  unsigned int v9; // r13d
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx

  if ( a1 <= 0x7F )
  {
    v6 = a2[1];
    v7 = v6 + 1;
    if ( (unsigned __int64)(v6 + 1) > a2[2] )
    {
LABEL_9:
      sub_C8D290((__int64)a2, a2 + 3, v7, 1u, a5, a6);
      v6 = a2[1];
    }
LABEL_6:
    *(_BYTE *)(*a2 + v6) = a1;
    ++a2[1];
    return;
  }
  if ( a1 <= 0x3FFF )
  {
    v8 = a2[1];
    v9 = (a1 >> 8) | 0xFFFFFF80;
    v10 = v8 + 1;
    if ( (unsigned __int64)(v8 + 1) <= a2[2] )
      goto LABEL_8;
    goto LABEL_15;
  }
  if ( a1 <= 0x1FFFFFFF )
  {
    v12 = a2[1];
    if ( (unsigned __int64)(v12 + 1) > a2[2] )
    {
      sub_C8D290((__int64)a2, a2 + 3, v12 + 1, 1u, a5, a6);
      v12 = a2[1];
    }
    *(_BYTE *)(*a2 + v12) = HIBYTE(a1) | 0xC0;
    v13 = a2[1];
    v14 = v13 + 1;
    v15 = v13 + 2;
    a2[1] = v14;
    if ( v15 > a2[2] )
    {
      sub_C8D290((__int64)a2, a2 + 3, v15, 1u, a5, a6);
      v14 = a2[1];
    }
    *(_BYTE *)(*a2 + v14) = BYTE2(a1);
    v16 = a2[1];
    v9 = a1 >> 8;
    v8 = v16 + 1;
    v10 = v16 + 2;
    a2[1] = v8;
    if ( v10 <= a2[2] )
    {
LABEL_8:
      *(_BYTE *)(*a2 + v8) = v9;
      v11 = a2[1];
      v6 = v11 + 1;
      v7 = v11 + 2;
      a2[1] = v6;
      if ( v7 > a2[2] )
        goto LABEL_9;
      goto LABEL_6;
    }
LABEL_15:
    sub_C8D290((__int64)a2, a2 + 3, v10, 1u, a5, a6);
    v8 = a2[1];
    goto LABEL_8;
  }
}
