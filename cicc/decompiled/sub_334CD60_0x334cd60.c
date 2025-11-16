// Function: sub_334CD60
// Address: 0x334cd60
//
__int64 __fastcall sub_334CD60(__int64 a1, unsigned __int64 **a2)
{
  __int64 v2; // rbx
  __int64 *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // r9
  unsigned int v7; // r15d
  __int64 v9; // r14
  unsigned __int64 *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 *v15; // rsi
  __int64 v16; // r10
  __int64 v17; // r14
  __int64 v18; // rax
  _QWORD *v19; // r15
  _QWORD *v20; // r14
  unsigned __int64 *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rsi
  int v26; // esi
  int v27; // r9d
  __int64 v29; // [rsp+10h] [rbp-220h]
  unsigned int v30; // [rsp+1Ch] [rbp-214h]
  __int64 *v31; // [rsp+20h] [rbp-210h]
  unsigned __int64 v32; // [rsp+28h] [rbp-208h]
  _BYTE v33[40]; // [rsp+30h] [rbp-200h] BYREF
  __int64 v34; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 *v35; // [rsp+60h] [rbp-1D0h]
  __int64 v36; // [rsp+70h] [rbp-1C0h] BYREF
  __int64 v37; // [rsp+78h] [rbp-1B8h]
  __int64 v38; // [rsp+80h] [rbp-1B0h] BYREF
  unsigned int v39; // [rsp+88h] [rbp-1A8h]
  char v40; // [rsp+200h] [rbp-30h] BYREF

  v2 = a1;
  sub_3754A80(v33, **(_QWORD **)(a1 + 592), *(_QWORD *)(a1 + 584), *a2);
  v3 = &v38;
  v36 = 0;
  v37 = 1;
  do
  {
    *v3 = 0;
    v3 += 3;
    *((_DWORD *)v3 - 4) = -1;
  }
  while ( v3 != (__int64 *)&v40 );
  v4 = *(_QWORD *)(a1 + 632);
  v5 = v34;
  v6 = (*(_QWORD *)(a1 + 640) - v4) >> 3;
  if ( (_DWORD)v6 )
  {
    v7 = v6 - 1;
    v31 = (__int64 *)(v34 + 40);
    while ( 1 )
    {
      v9 = *(_QWORD *)(v4 + 8LL * v7);
      if ( *(int *)(v9 + 24) >= 0 )
      {
        sub_3755B20(v33, v9, 0, 0, &v36);
        if ( (*(_BYTE *)(v9 + 32) & 1) == 0 )
          goto LABEL_6;
      }
      else
      {
        sub_37584B0(v33, v9, 0, 0, &v36);
        if ( (*(_BYTE *)(v9 + 32) & 1) == 0 )
          goto LABEL_6;
      }
      v10 = v35;
      v11 = *(_QWORD *)(*(_QWORD *)(v2 + 592) + 720LL);
      v12 = *(_QWORD *)(v11 + 696);
      v13 = *(unsigned int *)(v11 + 712);
      if ( !(_DWORD)v13 )
        goto LABEL_6;
      v14 = (v13 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v15 = (__int64 *)(v12 + 40LL * v14);
      v16 = *v15;
      if ( v9 != *v15 )
        break;
LABEL_12:
      if ( v15 == (__int64 *)(v12 + 40 * v13) )
        goto LABEL_6;
      v17 = v15[1];
      v18 = *((unsigned int *)v15 + 4);
      if ( v17 + 8 * v18 == v17 )
        goto LABEL_6;
      v30 = v7;
      v19 = (_QWORD *)v15[1];
      v20 = (_QWORD *)(v17 + 8 * v18);
      v29 = v2;
      v21 = v35;
      do
      {
        while ( 1 )
        {
          if ( !*(_BYTE *)(*v19 + 63LL) )
          {
            v22 = sub_37547E0(v33, *v19, &v36, v10);
            if ( v22 )
              break;
          }
          if ( v20 == ++v19 )
            goto LABEL_19;
        }
        v32 = v22;
        ++v19;
        sub_2E31040(v31, v22);
        v23 = *(_QWORD *)v32;
        v24 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v32 + 8) = v21;
        *(_QWORD *)v32 = v24 | v23 & 7;
        *(_QWORD *)(v24 + 8) = v32;
        *v21 = *v21 & 7 | v32;
      }
      while ( v20 != v19 );
LABEL_19:
      v2 = v29;
      v7 = v30 - 1;
      if ( !v30 )
      {
LABEL_20:
        v5 = v34;
        goto LABEL_21;
      }
LABEL_7:
      v4 = *(_QWORD *)(v2 + 632);
    }
    v26 = 1;
    while ( v16 != -4096 )
    {
      v27 = v26 + 1;
      v14 = (v13 - 1) & (v26 + v14);
      v15 = (__int64 *)(v12 + 40LL * v14);
      v16 = *v15;
      if ( v9 == *v15 )
        goto LABEL_12;
      v26 = v27;
    }
LABEL_6:
    if ( v7-- == 0 )
      goto LABEL_20;
    goto LABEL_7;
  }
LABEL_21:
  *a2 = v35;
  if ( (v37 & 1) == 0 )
    sub_C7D6A0(v38, 24LL * v39, 8);
  return v5;
}
