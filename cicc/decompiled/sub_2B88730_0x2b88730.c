// Function: sub_2B88730
// Address: 0x2b88730
//
void __fastcall sub_2B88730(__int64 a1, char ***a2)
{
  char **v2; // r13
  __int64 v3; // rax
  __int16 v4; // cx
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r9
  __int64 v11; // r13
  unsigned __int64 v12; // rsi
  _QWORD *v13; // rax
  int v14; // ecx
  _QWORD *v15; // rdx
  _QWORD *v16; // rax
  bool v17; // al
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  char v20; // dh
  char v21; // dl
  unsigned __int64 v22; // rsi
  __int64 v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  char **v25; // [rsp+18h] [rbp-48h]
  _QWORD v26[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = a2[52];
  v3 = sub_2B87C80(a1, a2);
  v4 = 0;
  v5 = v3;
  if ( *(_BYTE *)v3 == 84 )
  {
    v19 = sub_AA4FF0(*(_QWORD *)(v3 + 40));
    LOBYTE(v4) = 1;
    v6 = v19;
    v21 = 0;
    if ( v19 )
      v21 = v20;
    HIBYTE(v4) = v21;
    goto LABEL_31;
  }
  v6 = v3 + 24;
  if ( *((_DWORD *)a2 + 26) != 3
    && *((_DWORD *)a2 + 2)
    && ((v23 = *((unsigned int *)a2 + 2),
         v25 = *a2,
         v24 = (__int64)&(*a2)[v23],
         v16 = sub_2B0BF30(*a2, v24, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B099C0),
         v4 = 0,
         (_QWORD *)v24 == v16)
     || (v17 = sub_2B0D880(v25, v23, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B16010), v4 = 0, v17))
    || *(_BYTE *)(a1 + 1256)
    && *((_DWORD *)a2 + 50) >= *(_DWORD *)(a1 + 1252)
    && *((_DWORD *)a2 + 26) != 3
    && *(_BYTE *)a2[52] == 61 )
  {
LABEL_31:
    v7 = a1 + 3368;
    sub_A88F30(a1 + 3368, *(_QWORD *)(v5 + 40), v6, v4);
    goto LABEL_8;
  }
  v7 = a1 + 3368;
  v8 = sub_B46B10(v5, 0);
  sub_A88F30(a1 + 3368, *(_QWORD *)(v5 + 40), v8 + 24, 0);
LABEL_8:
  v9 = (__int64)v2[6];
  v26[0] = v9;
  if ( v9 && (sub_B96E90((__int64)v26, v9, 1), (v11 = v26[0]) != 0) )
  {
    v12 = *(unsigned int *)(a1 + 3376);
    v13 = *(_QWORD **)(a1 + 3368);
    v14 = *(_DWORD *)(a1 + 3376);
    v15 = &v13[2 * v12];
    if ( v13 != v15 )
    {
      while ( *(_DWORD *)v13 )
      {
        v13 += 2;
        if ( v15 == v13 )
          goto LABEL_24;
      }
      v13[1] = v26[0];
      goto LABEL_15;
    }
LABEL_24:
    v18 = *(unsigned int *)(a1 + 3380);
    if ( v12 >= v18 )
    {
      v22 = v12 + 1;
      if ( v18 < v22 )
      {
        sub_C8D5F0(v7, (const void *)(a1 + 3384), v22, 0x10u, a1 + 3384, v10);
        v15 = (_QWORD *)(*(_QWORD *)(a1 + 3368) + 16LL * *(unsigned int *)(a1 + 3376));
      }
      *v15 = 0;
      v15[1] = v11;
      v11 = v26[0];
      ++*(_DWORD *)(a1 + 3376);
    }
    else
    {
      if ( v15 )
      {
        *(_DWORD *)v15 = 0;
        v15[1] = v11;
        v11 = v26[0];
        v14 = *(_DWORD *)(a1 + 3376);
      }
      *(_DWORD *)(a1 + 3376) = v14 + 1;
    }
  }
  else
  {
    sub_93FB40(v7, 0);
    v11 = v26[0];
  }
  if ( v11 )
LABEL_15:
    sub_B91220((__int64)v26, v11);
}
