// Function: sub_1B9DCB0
// Address: 0x1b9dcb0
//
__int64 __fastcall sub_1B9DCB0(__int64 a1, __int64 *a2, unsigned int *a3)
{
  __int64 v6; // r14
  unsigned __int64 *v7; // rdx
  unsigned int v8; // r9d
  unsigned __int64 *v9; // r8
  unsigned __int64 *v10; // rdi
  unsigned __int64 *v11; // rax
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rcx
  unsigned __int64 *v14; // rdi
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rcx
  __int64 v18; // r15
  __int64 v19; // r14
  unsigned __int64 *v20; // rsi
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rax
  _QWORD *v24; // rdi
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 *v28; // [rsp+8h] [rbp-58h] BYREF
  unsigned __int64 *v29[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v30; // [rsp+20h] [rbp-40h]

  v6 = (__int64)a2;
  if ( !sub_13FC1A0(*(_QWORD *)(a1 + 8), (__int64)a2) )
  {
    v7 = *(unsigned __int64 **)(a1 + 352);
    v8 = *a3;
    v9 = (unsigned __int64 *)(a1 + 344);
    if ( !v7 )
      goto LABEL_26;
    v10 = (unsigned __int64 *)(a1 + 344);
    v11 = *(unsigned __int64 **)(a1 + 352);
    do
    {
      while ( 1 )
      {
        v12 = v11[2];
        v13 = v11[3];
        if ( v11[4] >= (unsigned __int64)a2 )
          break;
        v11 = (unsigned __int64 *)v11[3];
        if ( !v13 )
          goto LABEL_7;
      }
      v10 = v11;
      v11 = (unsigned __int64 *)v11[2];
    }
    while ( v12 );
LABEL_7:
    if ( v9 == v10 || v10[4] > (unsigned __int64)a2 )
      goto LABEL_26;
    v14 = (unsigned __int64 *)(a1 + 344);
    v15 = *(unsigned __int64 **)(a1 + 352);
    do
    {
      while ( 1 )
      {
        v16 = v15[2];
        v17 = v15[3];
        if ( v15[4] >= (unsigned __int64)a2 )
          break;
        v15 = (unsigned __int64 *)v15[3];
        if ( !v17 )
          goto LABEL_13;
      }
      v14 = v15;
      v15 = (unsigned __int64 *)v15[2];
    }
    while ( v16 );
LABEL_13:
    if ( v9 != v14 && v14[4] > (unsigned __int64)a2 )
      v14 = (unsigned __int64 *)(a1 + 344);
    v18 = a3[1];
    v19 = 48LL * v8;
    if ( *(_QWORD *)(*(_QWORD *)(v14[5] + v19) + 8 * v18) )
    {
      v28 = a2;
      v20 = (unsigned __int64 *)(a1 + 344);
      do
      {
        while ( 1 )
        {
          v21 = v7[2];
          v22 = v7[3];
          if ( v7[4] >= (unsigned __int64)a2 )
            break;
          v7 = (unsigned __int64 *)v7[3];
          if ( !v22 )
            goto LABEL_21;
        }
        v20 = v7;
        v7 = (unsigned __int64 *)v7[2];
      }
      while ( v21 );
LABEL_21:
      if ( v9 == v20 || v20[4] > (unsigned __int64)a2 )
      {
        v29[0] = (unsigned __int64 *)&v28;
        v20 = sub_1B99EB0((_QWORD *)(a1 + 336), v20, v29);
      }
      return *(_QWORD *)(*(_QWORD *)(v20[5] + v19) + 8 * v18);
    }
    else
    {
LABEL_26:
      v6 = sub_1B9C240((unsigned int *)a1, a2, v8);
      if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) == 16 )
      {
        v24 = *(_QWORD **)(a1 + 120);
        v25 = a3[1];
        v30 = 257;
        v26 = sub_1643350(v24);
        v27 = sub_159C470(v26, v25, 0);
        return sub_156D5F0((__int64 *)(a1 + 96), v6, v27, (__int64)v29);
      }
    }
  }
  return v6;
}
