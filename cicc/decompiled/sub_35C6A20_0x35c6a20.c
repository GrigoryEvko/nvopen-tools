// Function: sub_35C6A20
// Address: 0x35c6a20
//
bool __fastcall sub_35C6A20(_DWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  _QWORD *v4; // r14
  unsigned __int64 v6; // r13
  unsigned int v7; // ebx
  char v8; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r11
  __int64 v14; // r9
  __int64 v15; // r15
  unsigned int v16; // r13d
  __int64 v17; // rbx
  _QWORD *v18; // r8
  char v19; // r14
  __int64 v20; // r12
  unsigned int v21; // eax
  int v22; // edx
  char v23; // al
  __int64 i; // r13
  unsigned __int64 j; // rax
  __int64 v27; // r11
  __int64 v28; // r12
  __int64 v29; // r13
  unsigned int v30; // r15d
  __int64 v31; // rbx
  int v32; // edx
  char v33; // al
  unsigned __int64 v34; // [rsp+8h] [rbp-58h]
  _QWORD *v35; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+20h] [rbp-40h]
  __int64 v38; // [rsp+20h] [rbp-40h]
  _QWORD *v39; // [rsp+28h] [rbp-38h]
  unsigned int v40; // [rsp+28h] [rbp-38h]

  v3 = a2;
  v4 = a1;
  v6 = a3 + 48;
  v35 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)a1 + 16LL));
  sub_35C5BD0(a2, a3);
  v7 = a1[16];
  if ( v6 != *(_QWORD *)(a3 + 56) )
  {
    v8 = 0;
    while ( 1 )
    {
      while ( *(_QWORD *)(v3 + 32) != v6 )
        sub_35C5C00(v3);
      v9 = (_QWORD *)(*(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL);
      v10 = v9;
      if ( !v9 )
        BUG();
      v6 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
      v11 = *v9;
      if ( (v11 & 4) != 0 )
      {
        if ( v8 )
        {
          j = v6;
          goto LABEL_33;
        }
      }
      else if ( (*((_BYTE *)v10 + 44) & 4) != 0 )
      {
        for ( i = v11; ; i = *(_QWORD *)v6 )
        {
          v6 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v6 + 44) & 4) == 0 )
            break;
        }
        if ( v8 )
        {
          j = v6;
          if ( (*(_BYTE *)v6 & 4) == 0 )
          {
LABEL_31:
            for ( j = v6; (*(_BYTE *)(j + 44) & 8) != 0; j = *(_QWORD *)(j + 8) )
              ;
          }
LABEL_33:
          v27 = *(_QWORD *)(j + 8);
          v28 = *(_QWORD *)(v27 + 32);
          if ( v28 != v28 + 40LL * (*(_DWORD *)(v27 + 40) & 0xFFFFFF) )
          {
            v38 = *(_QWORD *)(j + 8);
            v34 = v6;
            v29 = v3;
            v30 = v7;
            v31 = v28 + 40LL * (*(_DWORD *)(v27 + 40) & 0xFFFFFF);
            do
            {
              if ( !*(_BYTE *)v28 )
              {
                v32 = *(_DWORD *)(v28 + 8);
                if ( v32 < 0 && v30 > (v32 & 0x7FFFFFFFu) )
                {
                  v33 = *(_BYTE *)(v28 + 4);
                  if ( (v33 & 1) == 0
                    && (v33 & 2) == 0
                    && ((*(_BYTE *)(v28 + 3) & 0x10) == 0 || (*(_DWORD *)v28 & 0xFFF00) != 0) )
                  {
                    v40 = sub_35C6920(v4, v29, v32, 1);
                    sub_2E8F280(v38, v40, v35, 0);
                    sub_35C59C0(v29, v40, -1, -1);
                  }
                }
              }
              v28 += 40;
            }
            while ( v31 != v28 );
            v7 = v30;
            v3 = v29;
            v6 = v34;
          }
        }
      }
      else if ( v8 )
      {
        goto LABEL_31;
      }
      v12 = *(_QWORD *)(v6 + 32);
      v8 = 0;
      v13 = v12 + 40LL * (*(_DWORD *)(v6 + 40) & 0xFFFFFF);
      if ( v12 == v13 )
        goto LABEL_23;
      v14 = v3;
      v15 = v6;
      v16 = v7;
      v17 = v12;
      v18 = v4;
      v19 = 0;
      v20 = v13;
      do
      {
        while ( 1 )
        {
          if ( *(_BYTE *)v17 )
            goto LABEL_15;
          v22 = *(_DWORD *)(v17 + 8);
          if ( v22 >= 0 || v16 <= (v22 & 0x7FFFFFFFu) )
            goto LABEL_15;
          v23 = *(_BYTE *)(v17 + 3) & 0x10;
          if ( (*(_BYTE *)(v17 + 4) & 1) != 0 || (*(_BYTE *)(v17 + 4) & 2) != 0 )
            break;
          if ( !v23 )
          {
            v19 = 1;
            goto LABEL_15;
          }
          if ( (*(_DWORD *)v17 & 0xFFF00) != 0 )
            v19 = 1;
LABEL_14:
          v37 = v14;
          v39 = v18;
          v21 = sub_35C6920(v18, v14, v22, 0);
          sub_2E8F690(v15, v21, v35, 0);
          v18 = v39;
          v14 = v37;
LABEL_15:
          v17 += 40;
          if ( v20 == v17 )
            goto LABEL_22;
        }
        if ( v23 )
          goto LABEL_14;
        v17 += 40;
      }
      while ( v20 != v17 );
LABEL_22:
      v8 = v19;
      v7 = v16;
      v4 = v18;
      v6 = v15;
      v3 = v14;
LABEL_23:
      if ( *(_QWORD *)(a3 + 56) == v6 )
        return *((_DWORD *)v4 + 16) != v7;
    }
  }
  return 0;
}
