// Function: sub_6AB110
// Address: 0x6ab110
//
__int64 __fastcall sub_6AB110(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  char i; // dl
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // r13
  bool v10; // zf
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // eax
  __int64 *v14; // r13
  _QWORD *v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rax
  int v22; // r14d
  __int16 v23; // bx
  __int64 v24; // r13
  __int64 v25; // rax
  char v26; // dl
  __int64 v27; // rax
  __int16 v28; // ax
  __int64 v29; // rdx
  FILE *v31; // r12
  unsigned int v32; // eax
  __int64 v33; // r12
  __int64 v34; // rdi
  _DWORD *v35; // r12
  __int64 v36; // [rsp+8h] [rbp-428h]
  __int64 v37; // [rsp+10h] [rbp-420h]
  __int64 v39; // [rsp+38h] [rbp-3F8h]
  int v40; // [rsp+40h] [rbp-3F0h]
  int v41; // [rsp+44h] [rbp-3ECh]
  int v42; // [rsp+48h] [rbp-3E8h]
  int v43; // [rsp+48h] [rbp-3E8h]
  __int64 v44; // [rsp+58h] [rbp-3D8h] BYREF
  __int64 v45; // [rsp+60h] [rbp-3D0h] BYREF
  __int64 v46; // [rsp+68h] [rbp-3C8h] BYREF
  _BYTE v47[32]; // [rsp+70h] [rbp-3C0h] BYREF
  char v48[160]; // [rsp+90h] [rbp-3A0h] BYREF
  _QWORD v49[2]; // [rsp+130h] [rbp-300h] BYREF
  char v50; // [rsp+140h] [rbp-2F0h]
  _BYTE v51[352]; // [rsp+290h] [rbp-1A0h] BYREF
  int v52; // [rsp+3F0h] [rbp-40h]
  __int16 v53; // [rsp+3F4h] [rbp-3Ch]

  sub_7ADF70(v47, 0);
  v45 = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(v47, 0, v1, v2);
  sub_7BE280(27, 125, 0, 0);
  v3 = qword_4F061C8;
  v4 = qword_4D03C50;
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  ++*(_QWORD *)(v4 + 40);
  ++*(_BYTE *)(v3 + 75);
  sub_6E1E00(5, v48, 0, 0);
  *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x20u;
  sub_69ED20((__int64)v49, 0, 0, 1);
  if ( dword_4F077C4 == 2 )
  {
    if ( !dword_4F077C0 || (_DWORD)qword_4F077B4 )
      goto LABEL_4;
  }
  else
  {
    if ( unk_4F07778 > 201709 )
    {
LABEL_3:
      sub_6F69D0(v49, 0);
      goto LABEL_4;
    }
    if ( !dword_4F077C0 )
    {
      if ( !(_DWORD)qword_4F077B4 )
        goto LABEL_4;
      goto LABEL_43;
    }
    if ( (_DWORD)qword_4F077B4 )
    {
LABEL_43:
      if ( qword_4F077A0 )
        goto LABEL_3;
      goto LABEL_4;
    }
  }
  if ( qword_4F077A8 )
    goto LABEL_3;
LABEL_4:
  sub_6F6C80(v49);
  if ( !v50 )
    goto LABEL_26;
  v5 = v49[0];
  v37 = v49[0];
  for ( i = *(_BYTE *)(v49[0] + 140LL); i == 12; i = *(_BYTE *)(v5 + 140) )
    v5 = *(_QWORD *)(v5 + 160);
  v42 = 0;
  if ( !i )
  {
LABEL_26:
    v42 = 1;
    v37 = sub_72C930(v49);
  }
  v7 = 253;
  v8 = 67;
  v9 = sub_6F6F40(v49, 0);
  v36 = v9;
  v41 = 0;
  v39 = 0;
  v10 = (unsigned int)sub_7BE280(67, 253, 0, 0) == 0;
  v13 = 1;
  if ( !v10 )
    v13 = v42;
  v14 = (__int64 *)(v9 + 16);
  v43 = v13;
  do
  {
    while ( 1 )
    {
      ++*(_BYTE *)(qword_4F061C8 + 63LL);
      v46 = *(_QWORD *)&dword_4F063F8;
      if ( word_4F06418[0] == 83 )
      {
        v40 = v39 == 0;
        if ( v41 )
        {
          v7 = (__int64)&dword_4F063F8;
          v8 = 2532;
          sub_6851C0(0x9E4u, &dword_4F063F8);
          v40 = 0;
          v43 = v41;
        }
        v44 = 0;
        sub_7B8B50(v8, v7, v11, v12);
        v41 = 1;
      }
      else
      {
        sub_65CD60(&v44);
        v26 = *(_BYTE *)(v44 + 140);
        if ( v26 == 12 )
        {
          v27 = v44;
          do
          {
            v27 = *(_QWORD *)(v27 + 160);
            v26 = *(_BYTE *)(v27 + 140);
          }
          while ( v26 == 12 );
        }
        if ( v26 )
        {
          v40 = sub_8D2530();
          if ( v40 )
          {
            v40 = sub_8D23B0(v44);
            if ( v40 )
            {
              v31 = (FILE *)v44;
              v32 = sub_67F240();
              v7 = (__int64)&v46;
              sub_685A50(v32, &v46, v31, 8u);
              v40 = 0;
              v43 = 1;
            }
            else if ( (unsigned int)sub_8DD010(v44) )
            {
              v7 = (__int64)&v46;
              sub_6851C0(0x9E5u, &v46);
              v43 = 1;
            }
            else
            {
              v33 = *(_QWORD *)(v36 + 16);
              if ( v33 )
              {
                while ( 1 )
                {
                  if ( *(_BYTE *)(v33 + 24) == 22 )
                  {
                    v7 = *(_QWORD *)(v33 + 56);
                    if ( v7 )
                    {
                      v34 = v44;
                      if ( v7 == v44 )
                        goto LABEL_61;
                      if ( (unsigned int)sub_8DED30(v44, v7, 1) )
                        break;
                    }
                  }
                  v33 = *(_QWORD *)(v33 + 16);
                  if ( !v33 )
                    goto LABEL_62;
                }
                v34 = v44;
LABEL_61:
                v7 = (__int64)&v46;
                sub_685360(0x9E6u, &v46, v34);
                v40 = 0;
                v43 = 1;
              }
              else
              {
LABEL_62:
                if ( v43 )
                {
                  v40 = 0;
                }
                else
                {
                  v7 = v44;
                  if ( v44 == v37 || (v40 = sub_8DED30(v37, v44, 1)) != 0 )
                  {
                    if ( v39 )
                    {
                      if ( *(_QWORD *)(v39 + 56) )
                      {
                        *(_QWORD *)v51 = 0;
                        *(_QWORD *)&v51[8] = 0;
                        v35 = sub_67D9D0(0xBD3u, &v46);
                        sub_686CA0(0xBD4u, v39 + 28, *(_QWORD *)(v39 + 56), v51);
                        v7 = (__int64)v51;
                        sub_67E370((__int64)v35, (const __m128i *)v51);
                        sub_685910((__int64)v35, (FILE *)v51);
                        sub_67E3D0(v51);
                        v43 = 1;
                      }
                      else
                      {
                        v7 = v39;
                        sub_6AB070((__int64)v47, v39);
                      }
                      v40 = 1;
                      v39 = *v14;
                      if ( *v14 )
                      {
                        v39 = 0;
                        v14 = (__int64 *)(*v14 + 16);
                      }
                    }
                    else
                    {
                      v40 = 1;
                    }
                  }
                }
              }
            }
          }
          else
          {
            v7 = (__int64)&v46;
            sub_6851C0(0x164u, &v46);
            v43 = 1;
          }
        }
        else
        {
          v40 = 0;
          v43 = 1;
        }
      }
      v15 = (_QWORD *)sub_726700(22);
      *v15 = sub_72CBE0(22, v7, v16, v17, v18, v19);
      v15[7] = v44;
      *(_QWORD *)((char *)v15 + 28) = v46;
      *v14 = (__int64)v15;
      --*(_BYTE *)(qword_4F061C8 + 63LL);
      if ( !(unsigned int)sub_7BE280(55, 53, 0, 0) )
        break;
      if ( ((v43 ^ 1) & v40) == 0 )
        goto LABEL_19;
      v14 = v15 + 2;
      v7 = (__int64)v51;
      memset(v51, 0, sizeof(v51));
      v52 = 0;
      v53 = 0;
      v51[67] = 1;
      v51[28] = 1;
      sub_7C6880(v47, v51);
      v8 = 67;
      v39 = (__int64)v15;
      if ( !(unsigned int)sub_7BE800(67) )
        goto LABEL_20;
    }
    v43 = 1;
LABEL_19:
    sub_69ED20((__int64)v49, 0, 0, 1);
    sub_6F6C80(v49);
    v7 = 0;
    v20 = sub_6F6F40(v49, 0);
    v8 = 67;
    v15[2] = v20;
    v14 = (__int64 *)(v20 + 16);
  }
  while ( (unsigned int)sub_7BE800(67) );
LABEL_20:
  if ( v43 )
  {
    if ( v39 )
    {
      sub_6AB070((__int64)v47, v39);
      v39 = 0;
    }
  }
  else if ( !v39 )
  {
    sub_685360(0x9E7u, &v45, v37);
    v43 = 1;
  }
  v21 = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 75LL);
  --*(_BYTE *)(v21 + 36);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  v22 = qword_4F063F0;
  v23 = WORD2(qword_4F063F0);
  sub_7BE280(28, 18, 0, 0);
  sub_6E2B30(28, 18);
  if ( v43 )
  {
    sub_6E6260(a1);
  }
  else
  {
    sub_6AB070((__int64)v47, v39);
    v24 = *(_QWORD *)(v39 + 16);
    *(_QWORD *)(v24 + 16) = 0;
    v25 = *(_QWORD *)(v39 + 16);
    if ( *(_BYTE *)(v25 + 24) == 2 )
    {
      sub_6E6A50(*(_QWORD *)(v25 + 56), a1);
      *(_QWORD *)(a1 + 288) = v24;
    }
    else
    {
      sub_6E7170(v24, a1);
    }
  }
  *(_DWORD *)(a1 + 68) = v45;
  v28 = WORD2(v45);
  *(_DWORD *)(a1 + 76) = v22;
  *(_WORD *)(a1 + 72) = v28;
  v29 = *(_QWORD *)(a1 + 68);
  *(_WORD *)(a1 + 80) = v23;
  *(_QWORD *)dword_4F07508 = v29;
  unk_4F061D8 = *(_QWORD *)(a1 + 76);
  return sub_6E3280(a1, &v45);
}
