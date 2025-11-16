// Function: sub_318B710
// Address: 0x318b710
//
__int64 __fastcall sub_318B710(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r9
  __int64 v17; // r13
  unsigned __int64 v18; // rsi
  _QWORD *v19; // rax
  int v20; // ecx
  _QWORD *v21; // rdx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rsi
  _QWORD v25[5]; // [rsp+8h] [rbp-28h] BYREF

  v9 = sub_371B390(&a7);
  v10 = *(_QWORD *)(v9 + 24);
  v11 = *(_QWORD *)(v9 + 16);
  if ( a8 != v11 + 48 )
  {
    v12 = sub_371B3B0(&a7, a8, a9);
    v13 = sub_318B5C0(v12);
    v14 = *(_QWORD *)(v13 + 40);
    *(_WORD *)(v10 + 448) = 0;
    *(_QWORD *)(v10 + 432) = v14;
    *(_QWORD *)(v10 + 440) = v13 + 24;
    v15 = *(_QWORD *)sub_B46C60(v13);
    v25[0] = v15;
    if ( v15 && (sub_B96E90((__int64)v25, v15, 1), (v17 = v25[0]) != 0) )
    {
      v18 = *(unsigned int *)(v10 + 392);
      v19 = *(_QWORD **)(v10 + 384);
      v20 = *(_DWORD *)(v10 + 392);
      v21 = &v19[2 * v18];
      if ( v19 != v21 )
      {
        while ( *(_DWORD *)v19 )
        {
          v19 += 2;
          if ( v21 == v19 )
            goto LABEL_14;
        }
        v19[1] = v25[0];
        goto LABEL_9;
      }
LABEL_14:
      v23 = *(unsigned int *)(v10 + 396);
      if ( v18 >= v23 )
      {
        v24 = v18 + 1;
        if ( v23 < v24 )
        {
          sub_C8D5F0(v10 + 384, (const void *)(v10 + 400), v24, 0x10u, v10 + 400, v16);
          v21 = (_QWORD *)(*(_QWORD *)(v10 + 384) + 16LL * *(unsigned int *)(v10 + 392));
        }
        *v21 = 0;
        v21[1] = v17;
        v17 = v25[0];
        ++*(_DWORD *)(v10 + 392);
      }
      else
      {
        if ( v21 )
        {
          *(_DWORD *)v21 = 0;
          v21[1] = v17;
          v17 = v25[0];
          v20 = *(_DWORD *)(v10 + 392);
        }
        *(_DWORD *)(v10 + 392) = v20 + 1;
      }
    }
    else
    {
      sub_93FB40(v10 + 384, 0);
      v17 = v25[0];
    }
    if ( !v17 )
      return v10 + 384;
LABEL_9:
    sub_B91220((__int64)v25, v17);
    return v10 + 384;
  }
  *(_QWORD *)(v10 + 440) = v11 + 48;
  *(_WORD *)(v10 + 448) = 0;
  *(_QWORD *)(v10 + 432) = v11;
  return v10 + 384;
}
