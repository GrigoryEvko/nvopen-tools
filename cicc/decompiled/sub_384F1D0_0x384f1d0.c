// Function: sub_384F1D0
// Address: 0x384f1d0
//
bool __fastcall sub_384F1D0(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v4; // rax
  __int64 v6; // r9
  unsigned int v7; // r11d
  __int64 *v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // esi
  _QWORD *v14; // r8
  __int64 v15; // r10
  __int64 v16; // rsi
  _QWORD *v17; // rdx
  __int64 v18; // rax
  int v20; // edx
  int v21; // r12d
  __int64 v22; // rdx
  int v23; // r8d
  int v24; // ebx

  v4 = *(unsigned int *)(a1 + 192);
  if ( !(_DWORD)v4 )
    return 0;
  v6 = *(_QWORD *)(a1 + 176);
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
  {
LABEL_3:
    if ( v8 != (__int64 *)(v6 + 16 * v4) )
    {
      v10 = v8[1];
      *a3 = v10;
      v11 = *(unsigned int *)(a1 + 224);
      v12 = *(_QWORD *)(a1 + 208);
      if ( (_DWORD)v11 )
      {
        v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v14 = (_QWORD *)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v10 == *v14 )
        {
LABEL_6:
          v16 = a1 + 200;
          v17 = (_QWORD *)(16 * v11 + v12);
          v18 = *(_QWORD *)(a1 + 200);
LABEL_7:
          a4[1] = v18;
          *a4 = v16;
          a4[2] = v14;
          a4[3] = v17;
          return v14 != (_QWORD *)(*(_QWORD *)(a1 + 208) + 16LL * *(unsigned int *)(a1 + 224));
        }
        v23 = 1;
        while ( v15 != -8 )
        {
          v24 = v23 + 1;
          v13 = (v11 - 1) & (v23 + v13);
          v14 = (_QWORD *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( v10 == *v14 )
            goto LABEL_6;
          v23 = v24;
        }
      }
      v16 = a1 + 200;
      v14 = (_QWORD *)(v12 + 16 * v11);
      v18 = *(_QWORD *)(a1 + 200);
      v17 = v14;
      goto LABEL_7;
    }
  }
  else
  {
    v20 = 1;
    while ( v9 != -8 )
    {
      v21 = v20 + 1;
      v22 = ((_DWORD)v4 - 1) & (v7 + v20);
      v7 = v22;
      v8 = (__int64 *)(v6 + 16 * v22);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v20 = v21;
    }
  }
  return 0;
}
