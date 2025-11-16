// Function: sub_11A07F0
// Address: 0x11a07f0
//
unsigned __int8 *__fastcall sub_11A07F0(__int64 a1, char a2)
{
  unsigned __int8 *v4; // rdi
  int v5; // ebx
  __int64 v6; // r15
  char v7; // al
  unsigned __int8 *v8; // r14
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  _DWORD *v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  _DWORD *v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rsi
  _DWORD *v21; // rax
  _DWORD *v22; // rsi
  __int64 v24; // [rsp+0h] [rbp-40h]
  __int64 v25; // [rsp+8h] [rbp-38h]

  v4 = **(unsigned __int8 ***)a1;
  v5 = *v4;
  v25 = **(_QWORD **)(a1 + 16);
  v24 = **(_QWORD **)(a1 + 8);
  if ( !a2 )
  {
    v25 = **(_QWORD **)(a1 + 8);
    v24 = **(_QWORD **)(a1 + 16);
  }
  v6 = *(_QWORD *)(v25 + 16);
  if ( v6 )
  {
    while ( 1 )
    {
      v7 = sub_B46D50(v4);
      v8 = *(unsigned __int8 **)(v6 + 24);
      v9 = *v8;
      if ( v7 )
      {
        if ( v5 != v9 )
          goto LABEL_6;
        v12 = *((_QWORD *)v8 - 8);
        v11 = *((_QWORD *)v8 - 4);
        if ( *(_BYTE *)v12 != 92 || *(_QWORD *)(v12 - 64) != v24 || !v11 || v25 != v11 )
        {
          if ( *(_BYTE *)v11 != 92 || *(_QWORD *)(v11 - 64) != v24 || v25 != v12 )
          {
LABEL_6:
            v6 = *(_QWORD *)(v6 + 8);
            if ( !v6 )
              return 0;
            goto LABEL_7;
          }
          goto LABEL_21;
        }
        v13 = *(_DWORD **)(v12 + 72);
        v14 = *(unsigned int *)(v12 + 80);
      }
      else
      {
        if ( !a2 )
        {
          if ( v5 != v9 )
            goto LABEL_6;
          v10 = *((_QWORD *)v8 - 8);
          if ( v25 != v10 )
            goto LABEL_6;
          if ( !v10 )
            goto LABEL_6;
          v11 = *((_QWORD *)v8 - 4);
          if ( *(_BYTE *)v11 != 92 || *(_QWORD *)(v11 - 64) != v24 )
            goto LABEL_6;
LABEL_21:
          v13 = *(_DWORD **)(v11 + 72);
          v14 = *(unsigned int *)(v11 + 80);
          goto LABEL_26;
        }
        if ( v5 != v9 )
          goto LABEL_6;
        v15 = *((_QWORD *)v8 - 8);
        if ( *(_BYTE *)v15 != 92 )
          goto LABEL_6;
        if ( *(_QWORD *)(v15 - 64) != v24 )
          goto LABEL_6;
        v13 = *(_DWORD **)(v15 + 72);
        v14 = *(unsigned int *)(v15 + 80);
        v16 = *((_QWORD *)v8 - 4);
        if ( v25 != v16 || !v16 )
          goto LABEL_6;
      }
LABEL_26:
      v17 = 4 * v14;
      v18 = &v13[(unsigned __int64)v17 / 4];
      v19 = v17 >> 2;
      v20 = v17 >> 4;
      if ( v20 )
      {
        v21 = v13;
        v22 = &v13[4 * v20];
        while ( (unsigned int)(*v21 + 1) <= 1 )
        {
          if ( (unsigned int)(v21[1] + 1) > 1 )
          {
            ++v21;
            goto LABEL_33;
          }
          if ( (unsigned int)(v21[2] + 1) > 1 )
          {
            v21 += 2;
            goto LABEL_33;
          }
          if ( (unsigned int)(v21[3] + 1) > 1 )
          {
            v21 += 3;
            goto LABEL_33;
          }
          v21 += 4;
          if ( v22 == v21 )
          {
            v19 = v18 - v21;
            goto LABEL_43;
          }
        }
        goto LABEL_33;
      }
      v21 = v13;
LABEL_43:
      if ( v19 == 2 )
        goto LABEL_50;
      if ( v19 != 3 )
      {
        if ( v19 != 1 )
          goto LABEL_34;
LABEL_46:
        if ( (unsigned int)(*v21 + 1) <= 1 )
          goto LABEL_34;
        goto LABEL_33;
      }
      if ( (unsigned int)(*v21 + 1) <= 1 )
      {
        ++v21;
LABEL_50:
        if ( (unsigned int)(*v21 + 1) <= 1 )
        {
          ++v21;
          goto LABEL_46;
        }
      }
LABEL_33:
      if ( v18 != v21 )
        goto LABEL_6;
LABEL_34:
      if ( *v13 == -1 )
        goto LABEL_6;
      if ( (unsigned __int8)sub_B19DB0(
                              *(_QWORD *)(*(_QWORD *)(a1 + 24) + 80LL),
                              *(_QWORD *)(v6 + 24),
                              **(_QWORD **)(a1 + 32)) )
        return v8;
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        return 0;
LABEL_7:
      v4 = **(unsigned __int8 ***)a1;
    }
  }
  return 0;
}
