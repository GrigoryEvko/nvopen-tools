// Function: sub_1EEB300
// Address: 0x1eeb300
//
bool __fastcall sub_1EEB300(_DWORD *a1, __int64 a2, __int64 a3, int a4, unsigned int a5, int a6)
{
  __int64 v6; // r15
  _QWORD *v7; // r14
  __int64 (*v8)(void); // rax
  unsigned __int64 v9; // r13
  unsigned int v10; // ebx
  char v11; // r12
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  __int64 i; // r13
  __int64 v16; // r8
  __int64 v17; // r11
  __int64 v18; // r9
  __int64 v19; // r15
  unsigned int v20; // r13d
  __int64 v21; // rbx
  _QWORD *v22; // r8
  char v23; // r14
  __int64 v24; // r12
  int v25; // eax
  __int64 v26; // r8
  unsigned __int64 v27; // r9
  int v28; // edx
  char v29; // al
  unsigned __int64 v31; // rax
  __int64 v32; // r11
  __int64 v33; // r12
  __int64 v34; // r13
  unsigned int v35; // r15d
  __int64 v36; // rbx
  int v37; // edx
  char v38; // al
  __int64 v39; // r8
  unsigned __int64 v40; // r9
  unsigned __int64 v41; // [rsp+8h] [rbp-58h]
  _QWORD *v42; // [rsp+10h] [rbp-50h]
  __int64 v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+20h] [rbp-40h]
  _QWORD *v46; // [rsp+28h] [rbp-38h]
  unsigned int v47; // [rsp+28h] [rbp-38h]

  v6 = a2;
  v7 = a1;
  v42 = 0;
  v8 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 112LL);
  if ( v8 != sub_1D00B10 )
    v42 = (_QWORD *)v8();
  v9 = a3 + 24;
  sub_1EEA050(a2, a3, a3, a4, a5, a6);
  v10 = a1[8];
  if ( a3 + 24 != *(_QWORD *)(a3 + 32) )
  {
    v11 = 0;
    while ( 1 )
    {
      v12 = (_QWORD *)(*(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL);
      v13 = v12;
      if ( !v12 )
        BUG();
      v9 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
      v14 = *v12;
      if ( (v14 & 4) == 0 && (*((_BYTE *)v13 + 46) & 4) != 0 )
      {
        for ( i = v14; ; i = *(_QWORD *)v9 )
        {
          v9 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v9 + 46) & 4) == 0 )
            break;
        }
      }
      while ( *(_QWORD *)(v6 + 32) != v9 )
        sub_1EEA510(v6);
      if ( v11 )
      {
        v31 = v9;
        if ( (*(_BYTE *)v9 & 4) == 0 && (*(_BYTE *)(v9 + 46) & 8) != 0 )
        {
          do
            v31 = *(_QWORD *)(v31 + 8);
          while ( (*(_BYTE *)(v31 + 46) & 8) != 0 );
        }
        v32 = *(_QWORD *)(v31 + 8);
        v33 = *(_QWORD *)(v32 + 32);
        if ( v33 + 40LL * *(unsigned int *)(v32 + 40) != v33 )
        {
          v45 = *(_QWORD *)(v31 + 8);
          v41 = v9;
          v34 = v6;
          v35 = v10;
          v36 = v33 + 40LL * *(unsigned int *)(v32 + 40);
          do
          {
            if ( !*(_BYTE *)v33 )
            {
              v37 = *(_DWORD *)(v33 + 8);
              if ( v37 < 0 && v35 > (v37 & 0x7FFFFFFFu) )
              {
                v38 = *(_BYTE *)(v33 + 4);
                if ( (v38 & 1) == 0
                  && (v38 & 2) == 0
                  && ((*(_BYTE *)(v33 + 3) & 0x10) == 0 || (*(_DWORD *)v33 & 0xFFF00) != 0) )
                {
                  v47 = sub_1EEB1E0(v7, v34, v37, 1);
                  sub_1E1AFE0(v45, v47, v42, 0, v39, v40);
                  sub_1EE9C20(v34, v47, -1);
                }
              }
            }
            v33 += 40;
          }
          while ( v36 != v33 );
          v10 = v35;
          v6 = v34;
          v9 = v41;
        }
      }
      v16 = *(_QWORD *)(v9 + 32);
      v17 = v16 + 40LL * *(unsigned int *)(v9 + 40);
      if ( v16 != v17 )
        break;
      v11 = 0;
LABEL_29:
      if ( *(_QWORD *)(a3 + 32) == v9 )
        return *((_DWORD *)v7 + 8) != v10;
    }
    v18 = v6;
    v19 = v9;
    v20 = v10;
    v21 = v16;
    v22 = v7;
    v23 = 0;
    v24 = v17;
    while ( 1 )
    {
      if ( *(_BYTE *)v21 )
        goto LABEL_21;
      v28 = *(_DWORD *)(v21 + 8);
      if ( v28 >= 0 || v20 <= (v28 & 0x7FFFFFFFu) )
        goto LABEL_21;
      v29 = *(_BYTE *)(v21 + 3) & 0x10;
      if ( (*(_BYTE *)(v21 + 4) & 1) != 0 || (*(_BYTE *)(v21 + 4) & 2) != 0 )
      {
        if ( v29 )
          goto LABEL_20;
        v21 += 40;
        if ( v24 == v21 )
        {
LABEL_28:
          v11 = v23;
          v10 = v20;
          v7 = v22;
          v9 = v19;
          v6 = v18;
          goto LABEL_29;
        }
      }
      else
      {
        if ( !v29 )
        {
          v23 = 1;
          goto LABEL_21;
        }
        if ( (*(_DWORD *)v21 & 0xFFF00) != 0 )
          v23 = 1;
LABEL_20:
        v44 = v18;
        v46 = v22;
        v25 = sub_1EEB1E0(v22, v18, v28, 0);
        sub_1E1B440(v19, v25, v42, 0, v26, v27);
        v22 = v46;
        v18 = v44;
LABEL_21:
        v21 += 40;
        if ( v24 == v21 )
          goto LABEL_28;
      }
    }
  }
  return 0;
}
