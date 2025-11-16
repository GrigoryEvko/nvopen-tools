// Function: sub_7BF3A0
// Address: 0x7bf3a0
//
__int64 __fastcall sub_7BF3A0(unsigned int a1, _DWORD *a2)
{
  _QWORD *v2; // r15
  __int64 v3; // r14
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  bool v11; // bl
  __int64 v12; // rax
  unsigned __int8 v13; // di
  _QWORD *v14; // r8
  char v15; // al
  int v16; // edx
  unsigned __int16 v17; // ax
  _QWORD *v18; // rax
  __int64 v19; // rax
  int v20; // eax
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v25; // rax
  __int16 v26; // ax
  __int64 v27; // rax
  char i; // dl
  char v29; // [rsp+7h] [rbp-69h]
  _QWORD *v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+20h] [rbp-50h]
  __int64 v33; // [rsp+20h] [rbp-50h]
  _QWORD *v34; // [rsp+20h] [rbp-50h]
  _QWORD *v35; // [rsp+20h] [rbp-50h]
  _QWORD *v36; // [rsp+20h] [rbp-50h]
  int v37; // [rsp+28h] [rbp-48h]
  _BOOL4 v39; // [rsp+34h] [rbp-3Ch] BYREF
  __int64 v40[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = 0;
  v3 = 0;
  v4 = 0;
  v5 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v29 = *(_BYTE *)(v5 + 12) & 1;
  *(_BYTE *)(v5 + 12) |= 1u;
  do
  {
    v37 = unk_4D03B6C;
    if ( word_4F06418[0] == 42 && dword_4F07770 )
      sub_7BC010(dword_4F07770);
    v6 = 0;
    if ( (unsigned int)sub_868D90(v40, 0, 0, 0, 1) )
    {
      while ( 1 )
      {
        v17 = word_4F06418[0];
        if ( word_4F06418[0] == 44 )
        {
          if ( !v4 )
          {
            if ( v3 )
            {
              v6 = (__int64)dword_4F07508;
              sub_6851C0(0x380u, dword_4F07508);
            }
            sub_867030(v40[0]);
            goto LABEL_44;
          }
          ++*(_BYTE *)(qword_4F061C8 + 75LL);
LABEL_27:
          v8 = a1;
          v18 = (_QWORD *)*v4;
          v9 = (__int64)v4;
          v16 = 1;
          *v4 = 0;
          v4 = v18;
          v11 = a1 == 0;
          goto LABEL_16;
        }
        v6 = a1;
        ++*(_BYTE *)(qword_4F061C8 + 75LL);
        v11 = a1 == 0;
        if ( v17 != 287 )
          break;
        v21 = sub_7BE320(v11, (unsigned int *)a1);
        v9 = (__int64)v21;
        if ( !v21 )
        {
          v6 = 0;
          sub_867630(v40[0], 0);
          v7 = (unsigned int)sub_866C00(v40[0]);
          --*(_BYTE *)(qword_4F061C8 + 75LL);
          goto LABEL_23;
        }
        v4 = (_QWORD *)*v21;
        v16 = 0;
        *v21 = 0;
LABEL_16:
        *(_BYTE *)(v9 + 24) = (2 * v11) | *(_BYTE *)(v9 + 24) & 0xFD;
        if ( !v3 )
          v3 = v9;
        if ( v2 )
          *v2 = v9;
        if ( v16 || (v6 = 0, v32 = v9, v19 = sub_867630(v40[0], 0), v9 = v32, (*(_QWORD *)(v32 + 16) = v19) == 0) )
        {
          v7 = 1;
          if ( !v4 )
            goto LABEL_30;
        }
        else
        {
          *(_BYTE *)(v32 + 24) |= 0x10u;
          v7 = 1;
          if ( !v4 )
          {
LABEL_30:
            v33 = v9;
            v20 = sub_866C00(v40[0]);
            v9 = v33;
            v7 = v20 != 0;
          }
        }
        v2 = (_QWORD *)v9;
        --*(_BYTE *)(qword_4F061C8 + 75LL);
LABEL_23:
        if ( !(_DWORD)v7 )
          goto LABEL_44;
      }
      if ( v4 )
        goto LABEL_27;
      if ( dword_4F077C4 == 2 )
      {
        if ( v17 != 1 || (word_4D04A10 & 0x200) == 0 )
        {
          v6 = 0;
          if ( !(unsigned int)sub_7C0F00(8404993, 0) )
            goto LABEL_11;
        }
      }
      else if ( v17 != 1 )
      {
        goto LABEL_11;
      }
      v6 = 0;
      v12 = sub_7BF130(1u, 0, &v39);
      if ( v12 )
      {
        if ( *(_BYTE *)(v12 + 80) == 19 && (unk_4D04A12 & 1) == 0 )
        {
          v6 = 0;
          v26 = sub_7BE840(0, 0);
          if ( v26 != 73 && v26 != 27 )
          {
            v13 = 2;
LABEL_12:
            v14 = sub_725090(v13);
            v15 = *((_BYTE *)v14 + 8);
            if ( v15 )
            {
              if ( v15 == 1 )
              {
                if ( a1 )
                {
                  v31 = v14;
                  v6 = (__int64)sub_724D80(0);
                  sub_6D6050(0, v6, 0, 0);
                  v9 = (__int64)v31;
                  v16 = 0;
                  v31[4] = v6;
                }
                else
                {
                  v14[4] = 0;
                  v36 = v14;
                  v25 = sub_6D6820(v37);
                  v9 = (__int64)v36;
                  v36[6] = v25;
                  if ( a2 )
                  {
                    v27 = *(_QWORD *)(v25 + 8);
                    for ( i = *(_BYTE *)(v27 + 140); i == 12; i = *(_BYTE *)(v27 + 140) )
                      v27 = *(_QWORD *)(v27 + 160);
                    if ( i )
                    {
                      v16 = 0;
                      v11 = 1;
                    }
                    else
                    {
                      v16 = 0;
                      v11 = 1;
                      *a2 = 1;
                    }
                  }
                  else
                  {
                    v16 = 0;
                    v4 = 0;
                    v11 = 1;
                  }
                }
              }
              else
              {
                v6 = (__int64)dword_4F07508;
                v35 = v14;
                v23 = sub_7C68A0(0, dword_4F07508, 0, 0);
                v9 = (__int64)v35;
                v16 = 0;
                v35[4] = v23;
              }
            }
            else
            {
              v6 = 0;
              v34 = v14;
              v22 = sub_65CFF0(&v39, 0);
              v9 = (__int64)v34;
              v34[4] = v22;
              if ( !a1 )
                *((_BYTE *)v34 + 24) = (v39 << 7) | v34[3] & 0x7F;
              v16 = 0;
            }
            goto LABEL_16;
          }
        }
      }
LABEL_11:
      v13 = (unsigned int)sub_679C10(0x85u) == 0;
      goto LABEL_12;
    }
LABEL_44:
    ;
  }
  while ( (unsigned int)sub_7BE800(0x43u, (unsigned int *)v6, v7, v8, v9, v10) );
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) = v29
                                                            | *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12)
                                                            & 0xFE;
  return v3;
}
