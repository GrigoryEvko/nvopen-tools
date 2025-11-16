// Function: sub_6A9320
// Address: 0x6a9320
//
__int64 __fastcall sub_6A9320(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, int a6, __int64 a7)
{
  unsigned int v7; // r15d
  _QWORD *v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  int v15; // eax
  __int64 v16; // r12
  bool v17; // zf
  __int64 v18; // r12
  __int64 v19; // rdx
  __int64 v21; // r14
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v33; // [rsp+18h] [rbp-68h]
  __int16 v34; // [rsp+26h] [rbp-5Ah]
  int v35; // [rsp+28h] [rbp-58h]
  __int64 v36; // [rsp+28h] [rbp-58h]
  int v37; // [rsp+3Ch] [rbp-44h] BYREF
  __int64 v38; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v39[7]; // [rsp+48h] [rbp-38h] BYREF

  v7 = a4;
  v10 = a1;
  v11 = a7;
  v37 = 0;
  if ( a1 )
  {
    v36 = *a1;
    v12 = sub_6E3DA0(*a1, 0);
    v13 = v36;
    v39[0] = *(_QWORD *)(v12 + 356);
    v35 = *(_DWORD *)(v36 + 44);
    v34 = *(_WORD *)(v13 + 48);
    a1[2] = *(_QWORD *)(v13 + 64);
  }
  else
  {
    v39[0] = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(0, a2, a3, a4);
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
  }
  v14 = sub_6A8F30(a1, (FILE *)v7);
  v33 = v14;
  v15 = v37 | (*(_BYTE *)(v14 + 24) == 0);
  v37 = v15;
  if ( a5 )
  {
    if ( a6 )
    {
      if ( a1 )
      {
        v21 = a1[2];
        if ( v21 )
        {
          v22 = v15;
          do
          {
            if ( v22 )
            {
              v10 = a1;
              v11 = a7;
              goto LABEL_13;
            }
            if ( (*(_BYTE *)(v21 + 26) & 4) != 0 )
            {
              v24 = sub_6E3DA0(v21, 0);
              if ( (unsigned int)sub_869530(
                                   *(_QWORD *)(v24 + 128),
                                   a1[4],
                                   a1[3],
                                   (unsigned int)&v38,
                                   *((_DWORD *)a1 + 10),
                                   a1[6],
                                   (__int64)&v37) )
              {
                do
                {
                  a1[2] = v21;
                  v25 = v14;
                  v14 = sub_6A8F30(a1, (FILE *)a5);
                  v17 = *(_BYTE *)(v14 + 24) == 0;
                  *(_QWORD *)(v25 + 16) = v14;
                  v37 |= v17;
                  sub_867630(v38, 0);
                }
                while ( (unsigned int)sub_866C00(v38) );
              }
              v22 = v37;
            }
            else
            {
              v23 = sub_6A8F30(a1, (FILE *)a5);
              v17 = *(_BYTE *)(v23 + 24) == 0;
              *(_QWORD *)(v14 + 16) = v23;
              v14 = v23;
              v22 = v37 | v17;
              v37 = v22;
            }
            v21 = *(_QWORD *)(v21 + 16);
          }
          while ( v21 );
          v10 = a1;
          v15 = v22;
          v11 = a7;
        }
      }
      else if ( word_4F06418[0] == 67 )
      {
        v27 = v14;
        do
        {
          sub_7BE280(67, 253, 0, 0);
          if ( (unsigned int)sub_869470(&v38) )
          {
            do
            {
              v28 = v27;
              v27 = sub_6A8F30(0, (FILE *)a5);
              v17 = *(_BYTE *)(v27 + 24) == 0;
              *(_QWORD *)(v28 + 16) = v27;
              v37 |= v17;
              v29 = sub_867630(v38, 0);
              if ( v29 )
              {
                if ( *(_BYTE *)(v27 + 24) )
                {
                  *(_BYTE *)(v27 + 26) |= 4u;
                  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
                    *(_QWORD *)(*(_QWORD *)(v27 + 80) + 128LL) = v29;
                }
              }
            }
            while ( (unsigned int)sub_866C00(v38) );
          }
        }
        while ( word_4F06418[0] == 67 );
        v10 = 0;
        v15 = v37;
      }
    }
    else
    {
      if ( a1 )
      {
        v16 = sub_6A8F30(a1, (FILE *)a5);
        v17 = *(_BYTE *)(v16 + 24) == 0;
        *(_QWORD *)(v14 + 16) = v16;
        v15 = v37 | v17;
        v37 = v15;
        if ( (_BYTE)a2 != 76 )
          goto LABEL_7;
        v30 = sub_6A8F30(a1, (FILE *)4);
        *(_QWORD *)(v16 + 16) = v30;
        v37 |= *(_BYTE *)(v30 + 24) == 0;
        goto LABEL_40;
      }
      sub_7BE280(67, 253, 0, 0);
      v16 = sub_6A8F30(0, (FILE *)a5);
      v17 = *(_BYTE *)(v16 + 24) == 0;
      *(_QWORD *)(v14 + 16) = v16;
      v15 = v37 | v17;
      v37 = v15;
      if ( (_BYTE)a2 == 76 )
      {
        sub_7BE280(67, 253, 0, 0);
        v31 = sub_6A8F30(0, (FILE *)4);
        *(_QWORD *)(v16 + 16) = v31;
        v37 |= *(_BYTE *)(v31 + 24) == 0;
        sub_7BE280(67, 253, 0, 0);
        v30 = *(_QWORD *)(v16 + 16);
LABEL_40:
        *(_QWORD *)(v30 + 16) = sub_6A8F30(a1, (FILE *)4);
        v15 = v37 | (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v16 + 16) + 16LL) + 24LL) == 0);
        v37 = v15;
      }
    }
  }
LABEL_7:
  if ( v15 )
  {
LABEL_13:
    sub_6E6260(v11);
    if ( !v10 )
    {
LABEL_28:
      v26 = qword_4D03C50;
      v35 = qword_4F063F0;
      v34 = WORD2(qword_4F063F0);
      --*(_BYTE *)(qword_4F061C8 + 36LL);
      --*(_QWORD *)(v26 + 40);
      sub_7BE280(28, 18, 0, 0);
      goto LABEL_15;
    }
    *((_BYTE *)v10 + 56) = 1;
  }
  else
  {
    v18 = sub_726700(23);
    *(_QWORD *)v18 = a3;
    *(_QWORD *)(v18 + 28) = v39[0];
    *(_BYTE *)(v18 + 56) = a2;
    *(_QWORD *)(v18 + 64) = v33;
    sub_6E3AC0(v18, v39, 0, 0);
    sub_6E2E50(2, v11);
    sub_7197C0(v18, v11 + 144, *(_BYTE *)(qword_4D03C50 + 16LL) != 0, v39, &v38);
    if ( (_DWORD)v38 )
    {
      sub_6E70E0(v18, v11);
    }
    else
    {
      v19 = *(_QWORD *)(v11 + 272);
      *(_BYTE *)(v11 + 17) = 2;
      *(_QWORD *)v11 = v19;
    }
    if ( !v10 )
      goto LABEL_28;
  }
LABEL_15:
  *(_DWORD *)(v11 + 68) = v39[0];
  *(_WORD *)(v11 + 72) = WORD2(v39[0]);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v11 + 68);
  *(_DWORD *)(v11 + 76) = v35;
  *(_WORD *)(v11 + 80) = v34;
  unk_4F061D8 = *(_QWORD *)(v11 + 76);
  return sub_6E3280(v11, v39);
}
