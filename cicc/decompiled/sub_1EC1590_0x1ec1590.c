// Function: sub_1EC1590
// Address: 0x1ec1590
//
__int64 __fastcall sub_1EC1590(__int64 a1, char **a2, __int64 a3)
{
  unsigned int v4; // eax
  __int64 v5; // r8
  unsigned int v6; // r12d
  unsigned __int64 v7; // rax
  unsigned int v8; // r15d
  __int64 v9; // r14
  unsigned int v10; // r9d
  __int64 v11; // rdx
  int *v12; // rdx
  int v13; // eax
  __int64 (*v14)(void); // rax
  unsigned __int64 v15; // rdx
  unsigned int v16; // r12d
  char v17; // r8
  unsigned int v18; // eax
  unsigned int v19; // eax
  char *v20; // rsi
  char *v21; // rsi
  __int64 v23; // r10
  _QWORD *v24; // rcx
  _QWORD *v25; // rax
  char v26; // cl
  __int64 v27; // rax
  __int64 *v28; // rsi
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  char v32; // al
  unsigned __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  char v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+28h] [rbp-38h] BYREF

  v4 = sub_1DB4D20(a3);
  v5 = a3;
  v6 = v4;
  v7 = *(unsigned int *)(a1 + 928);
  v8 = *(_DWORD *)(a3 + 112);
  v9 = v8 & 0x7FFFFFFF;
  v10 = v9 + 1;
  if ( (int)v9 + 1 <= (unsigned int)v7 )
    goto LABEL_2;
  v23 = v10;
  if ( v10 < v7 )
  {
    *(_DWORD *)(a1 + 928) = v10;
    goto LABEL_2;
  }
  if ( v10 <= v7 )
  {
LABEL_2:
    v11 = *(_QWORD *)(a1 + 920);
    goto LABEL_3;
  }
  if ( v10 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
  {
    v34 = a3;
    v39 = v10;
    sub_16CD150(a1 + 920, (const void *)(a1 + 936), v10, 8, v5, v10);
    v7 = *(unsigned int *)(a1 + 928);
    v5 = v34;
    v10 = v9 + 1;
    v23 = v39;
  }
  v11 = *(_QWORD *)(a1 + 920);
  v24 = (_QWORD *)(v11 + 8 * v23);
  v25 = (_QWORD *)(v11 + 8 * v7);
  if ( v24 != v25 )
  {
    do
    {
      if ( v25 )
        *v25 = *(_QWORD *)(a1 + 936);
      ++v25;
    }
    while ( v24 != v25 );
    v11 = *(_QWORD *)(a1 + 920);
  }
  *(_DWORD *)(a1 + 928) = v10;
LABEL_3:
  v12 = (int *)(v11 + 8 * v9);
  v13 = *v12;
  if ( !*v12 )
  {
    *v12 = 1;
    v13 = *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v9);
  }
  if ( v13 != 2 )
  {
    if ( v13 == 5 )
    {
      v6 = dword_4FC91D0++;
      goto LABEL_13;
    }
    v14 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 696) + 264LL);
    if ( v14 == sub_1EBAF60 )
    {
      v15 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL) + 16 * v9) & 0xFFFFFFFFFFFFFFF8LL;
    }
    else
    {
      v36 = v5;
      v32 = v14();
      v5 = v36;
      v26 = v32;
      v15 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL) + 16 * v9) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v32 )
      {
        if ( *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v9) != 1 )
          goto LABEL_10;
        goto LABEL_32;
      }
    }
    if ( v6 >> 4 > 2 * (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v15 + 20LL)
      || *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v9) != 1 )
    {
      goto LABEL_10;
    }
    v26 = 0;
LABEL_32:
    if ( *(_DWORD *)(v5 + 8) )
    {
      v33 = v15;
      v35 = v26;
      v38 = v5;
      if ( sub_1DBCA20(*(_QWORD *)(a1 + 264), v5) )
      {
        v27 = *(_QWORD *)(a1 + 800);
        v28 = *(__int64 **)v38;
        if ( v35 )
        {
          v30 = *(_QWORD *)(v27 + 344);
          v31 = v28[3 * *(unsigned int *)(v38 + 8) - 2] & 0xFFFFFFFFFFFFFFF8LL;
        }
        else
        {
          v29 = *(_QWORD *)(v27 + 336);
          v30 = *v28;
          v31 = v29 & 0xFFFFFFFFFFFFFFF8LL;
        }
        v16 = ((unsigned int)(*(_DWORD *)(v31 + 24) - *(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 24)) >> 2)
            | (*(unsigned __int8 *)(v33 + 28) << 24);
LABEL_11:
        v17 = sub_1F5BE90(*(_QWORD *)(a1 + 256), v8);
        v18 = v16;
        v6 = v16 | 0xC0000000;
        v19 = v18 | 0x80000000;
        if ( !v17 )
          v6 = v19;
        goto LABEL_13;
      }
    }
LABEL_10:
    v16 = v6 + 0x20000000;
    goto LABEL_11;
  }
LABEL_13:
  LODWORD(v40) = v6;
  v20 = a2[1];
  HIDWORD(v40) = ~v8;
  if ( v20 == a2[2] )
  {
    sub_1E0C2B0(a2, v20, &v40);
    v21 = a2[1];
  }
  else
  {
    if ( v20 )
    {
      *(_QWORD *)v20 = v40;
      v20 = a2[1];
    }
    v21 = v20 + 8;
    a2[1] = v21;
  }
  return sub_1EBB580((__int64)*a2, ((v21 - *a2) >> 3) - 1, 0, *((_QWORD *)v21 - 1));
}
