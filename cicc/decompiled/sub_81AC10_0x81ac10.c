// Function: sub_81AC10
// Address: 0x81ac10
//
char *__fastcall sub_81AC10(unsigned __int64 a1)
{
  unsigned __int64 v1; // rsi
  unsigned __int64 v2; // r8
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdi
  _BYTE *v5; // rax
  _QWORD *v6; // rdx
  size_t v7; // r13
  void *v8; // rcx
  char *v9; // rax
  _BYTE *v10; // r13
  _QWORD *v11; // r15
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r8
  void *v14; // r12
  void *v15; // r14
  signed __int64 v16; // rbx
  void *v17; // rdi
  _QWORD *v18; // r12
  __int64 v19; // r14
  unsigned __int64 v20; // rax
  _QWORD *v22; // r12
  __int64 v23; // rdx
  unsigned __int64 v24; // r15
  _BYTE *v25; // rbx
  _BYTE *v26; // r9
  char v27; // r14
  _BYTE *v28; // r10
  _BYTE *v29; // r10
  unsigned __int64 v30; // r9
  char v31; // si
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r14
  void *v35; // r12
  __int64 v36; // rbx
  __int64 v37; // rbx
  _QWORD *v38; // [rsp+0h] [rbp-80h]
  unsigned __int64 v39; // [rsp+8h] [rbp-78h]
  __int64 v40; // [rsp+10h] [rbp-70h]
  _BYTE *v41; // [rsp+18h] [rbp-68h]
  unsigned __int64 v42; // [rsp+20h] [rbp-60h]
  _BYTE *v43; // [rsp+20h] [rbp-60h]
  _BYTE *v44; // [rsp+20h] [rbp-60h]
  _QWORD *v45; // [rsp+28h] [rbp-58h]
  unsigned __int64 v46; // [rsp+28h] [rbp-58h]
  __int64 v47; // [rsp+28h] [rbp-58h]
  _BYTE *v48; // [rsp+30h] [rbp-50h]
  _QWORD *v49; // [rsp+30h] [rbp-50h]
  unsigned __int64 v50; // [rsp+30h] [rbp-50h]
  char *v51; // [rsp+38h] [rbp-48h]
  __int64 v52[7]; // [rsp+48h] [rbp-38h] BYREF

  v1 = qword_4F195A8 - (_QWORD)qword_4F195B0;
  if ( ~((_BYTE *)qword_4F195A0 - (_BYTE *)qword_4F195B0) <= a1 )
    goto LABEL_62;
  v2 = a1 + (_BYTE *)qword_4F195A0 - (_BYTE *)qword_4F195B0;
  v3 = a1 / 0xA - (qword_4F195A8 - (_QWORD)qword_4F195A0) + a1;
  if ( qword_4F19590 > v3 && qword_4F19590 > v1 >> 2 )
  {
    v4 = v1 + 1;
    v40 = qword_4F195A8 - (_QWORD)qword_4F195B0;
    v39 = v1 + 1;
    goto LABEL_10;
  }
  if ( v1 >= v3 )
    v3 = qword_4F195A8 - (_QWORD)qword_4F195B0;
  v40 = v1 + v3;
  v4 = v1 + v3 + 1;
  if ( v4 < v2 )
  {
    if ( v3 )
LABEL_62:
      sub_685240(0x6D9u);
  }
  v39 = v1 + 1;
LABEL_10:
  v5 = (_BYTE *)sub_822B10(v4);
  v7 = qword_4F19598;
  v41 = v5;
  if ( qword_4F19598 )
  {
    v35 = qword_4F195B0;
    v36 = (__int64)v5;
    memcpy(v5, qword_4F195B0, qword_4F19598);
    sub_81A600((unsigned __int64)v35, (__int64)((__int64)v35 + v7) - 1, v36, 0);
    v7 = qword_4F19598;
  }
  v8 = qword_4F195B0;
  v9 = (char *)qword_4F195B0 + v7;
  v10 = &v41[v7];
  v51 = v9;
  v11 = (_QWORD *)qword_4F06440;
  if ( qword_4F06440 )
  {
    while ( 1 )
    {
      v12 = v11[2];
      if ( v12 && v12 < (unsigned __int64)v51 && v12 >= (unsigned __int64)v8 )
      {
        sub_7AED90((__int64)v11);
        v11[2] = &v41[v11[2]] - (_BYTE *)qword_4F195B0;
        sub_7AED40((__int64)v11);
        v8 = qword_4F195B0;
      }
      v13 = v11[7];
      LOBYTE(v6) = (unsigned __int64)v8 <= v13;
      if ( v13 < (unsigned __int64)v51 && (unsigned __int64)v8 <= v13 )
      {
        v6 = (_QWORD *)v11[13];
        v11[7] = &v41[v13 - (_QWORD)v8];
        v11[8] = &v41[v11[8] - (_QWORD)v8];
        if ( !v6 )
          goto LABEL_16;
        do
        {
          v6[1] = &v41[v6[1] - (_QWORD)v8];
          v6 = (_QWORD *)*v6;
        }
        while ( v6 );
        v11 = (_QWORD *)*v11;
        if ( !v11 )
          break;
      }
      else
      {
        if ( v13 >= (unsigned __int64)v51 && v13 < (unsigned __int64)qword_4F195A0 )
        {
          v11[7] = v10;
          v22 = (_QWORD *)v11[13];
          v23 = (__int64)v10;
          v38 = v11;
          v24 = v13;
          while ( 2 )
          {
            v25 = (_BYTE *)v24;
            do
            {
              v26 = v25;
              v27 = *v25;
              v28 = v10;
              ++v25;
              *v10++ = v27;
            }
            while ( v27 != 10 && v27 != 3 );
            if ( (unsigned __int64)v25 <= v24 + 1 || *(v25 - 2) )
            {
              v43 = v28;
              v46 = (unsigned __int64)v26;
              v49 = (_QWORD *)v23;
              sub_81A600(v24, (__int64)v25, v23, 0);
              v6 = v49;
              v30 = v46;
              LOBYTE(v8) = v27 == 10;
              v29 = v43;
              if ( v22 )
              {
                v31 = 0;
                goto LABEL_46;
              }
              if ( v27 == 10 )
                goto LABEL_54;
            }
            else
            {
              v48 = v28;
              v42 = (unsigned __int64)v26;
              v45 = (_QWORD *)v23;
              sub_81A600(v24, (__int64)v25, v23, 0);
              v29 = v48;
              if ( !v22 )
              {
                v31 = 1;
                goto LABEL_48;
              }
              v6 = v45;
              v30 = v42;
              v8 = 0;
              v31 = 1;
              do
              {
LABEL_46:
                v32 = v22[1];
                if ( v32 >= (unsigned __int64)v25 )
                  break;
                v22[1] = (char *)v6 + v32 - v24;
                v22 = (_QWORD *)*v22;
              }
              while ( v22 );
              if ( !(_BYTE)v8 )
              {
LABEL_48:
                if ( v27 == 3 && v31 )
                {
                  v11 = v38;
                  v38[8] = v29 - 1;
                  break;
                }
                goto LABEL_50;
              }
LABEL_54:
              v44 = v29;
              v47 = (__int64)v6;
              v50 = v30;
              if ( (unsigned int)sub_7AF220(v30) )
              {
                v33 = sub_7AF1D0(v50);
                v34 = v33;
                if ( *(_QWORD *)(v33 + 56) == v33 + 51 )
                {
                  sub_7AED90(v33);
                  *(_QWORD *)(v34 + 16) = v44;
                  sub_7AED40(v34);
                  if ( !(unsigned int)sub_81A600(v24, v50 + *(_QWORD *)(v34 + 32), v47, 0) && *(char *)(v34 + 48) >= 0 )
                  {
                    v37 = *(_QWORD *)(v34 + 32);
                    *(_QWORD *)(v34 + 32) = 1;
                    v25 = (_BYTE *)(v50 + v37);
                  }
                }
                else
                {
                  v25 = (_BYTE *)(v50 + *(_QWORD *)(v33 + 32));
                  v10 = sub_818F90(v33, v10);
                }
              }
            }
LABEL_50:
            v23 = (__int64)v10;
            v24 = (unsigned __int64)v25;
            continue;
          }
        }
LABEL_16:
        v11 = (_QWORD *)*v11;
        if ( !v11 )
          break;
      }
      v8 = qword_4F195B0;
    }
  }
  v14 = qword_4F19588;
  qword_4F19598 = v10 - v41;
  if ( qword_4F19588 )
  {
    v15 = qword_4F195A0;
    v16 = (_BYTE *)qword_4F195A0 - (_BYTE *)qword_4F19588;
    memcpy(v10, qword_4F19588, (_BYTE *)qword_4F195A0 - (_BYTE *)qword_4F19588);
    sub_81A600((unsigned __int64)v14, (__int64)v15, (__int64)v10, 0);
    qword_4F19588 = v10;
    v10 += v16;
  }
  v17 = qword_4F195B0;
  v18 = (_QWORD *)qword_4F06440;
  if ( qword_4F06440 )
  {
    while ( 1 )
    {
      v19 = (__int64)v18;
      v18 = (_QWORD *)*v18;
      v20 = *(_QWORD *)(v19 + 16);
      if ( v20 < (unsigned __int64)v17 || v20 >= (unsigned __int64)qword_4F195A0 )
        goto LABEL_31;
      if ( (*(_BYTE *)(v19 + 48) & 0x20) != 0 )
      {
        sub_7AED90(v19);
        *(_QWORD *)(v19 + 16) = 0;
        v17 = qword_4F195B0;
        *(_QWORD *)(v19 + 32) = 0;
LABEL_31:
        if ( !v18 )
          break;
      }
      else
      {
        v52[0] = v19;
        sub_7AEF90(v19);
        sub_7AEF30((__int64)v52);
        v17 = qword_4F195B0;
        if ( !v18 )
          break;
      }
    }
  }
  sub_822B90(v17, v39, v6, v8);
  qword_4F195A0 = v10;
  qword_4F19590 = 0;
  qword_4F195B0 = v41;
  qword_4F195A8 = (__int64)&v41[v40];
  return &v41[v40];
}
