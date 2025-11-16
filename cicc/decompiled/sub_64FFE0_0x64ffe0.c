// Function: sub_64FFE0
// Address: 0x64ffe0
//
__int64 __fastcall sub_64FFE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  char v7; // al
  bool v8; // bl
  bool v9; // cf
  bool v10; // zf
  __int64 v11; // rcx
  _QWORD *v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  _QWORD *v18; // r14
  char v19; // al
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // rdi
  unsigned __int64 v25; // rcx
  __int64 v26; // r15
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // rax
  char v29; // [rsp+0h] [rbp-60h]
  unsigned __int64 v30; // [rsp+0h] [rbp-60h]
  unsigned __int64 v31; // [rsp+8h] [rbp-58h]
  unsigned int v32; // [rsp+14h] [rbp-4Ch]
  __int64 v33; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v34[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = a1;
  v33 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v7 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4);
  v8 = v7 == 2 || ((v7 - 15) & 0xFD) == 0;
  if ( v8 )
  {
    sub_854AB0();
    v34[0] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077C4 == 2 || unk_4F07778 <= 202310 )
    {
      v12 = (_QWORD *)sub_727060();
LABEL_10:
      v14 = unk_4D03B98 + 176LL * unk_4D03B90;
      v15 = sub_727640();
      *(_BYTE *)(v15 + 8) = 69;
      *(_QWORD *)(v15 + 16) = v12;
      *(_QWORD *)v15 = **(_QWORD **)(v14 + 136);
      **(_QWORD **)(v14 + 136) = v15;
      goto LABEL_11;
    }
LABEL_4:
    if ( qword_4D04A00 )
    {
      v9 = *(_QWORD *)(qword_4D04A00 + 16) < 0xEu;
      v10 = *(_QWORD *)(qword_4D04A00 + 16) == 14;
      if ( *(_QWORD *)(qword_4D04A00 + 16) == 14 )
      {
        a2 = *(_QWORD *)(qword_4D04A00 + 8);
        v23 = 14;
        a1 = (__int64)"_Static_assert";
        do
        {
          if ( !v23 )
            break;
          v9 = *(_BYTE *)a2 < *(_BYTE *)a1;
          v10 = *(_BYTE *)a2++ == *(_BYTE *)a1++;
          --v23;
        }
        while ( v10 );
        if ( (!v9 && !v10) == v9 )
        {
          a1 = dword_4F063F8;
          if ( !(unsigned int)sub_729F80(dword_4F063F8) )
          {
            sub_684AA0(4 - ((unsigned int)(dword_4D04964 == 0) - 1), 3293, &dword_4F063F8);
            a2 = 1;
            a1 = 3293;
            sub_67D850(3293, 1, 0);
          }
        }
      }
    }
    v12 = (_QWORD *)sub_727060();
    if ( !v8 )
    {
      v13 = dword_4F04C3C;
      if ( dword_4F04C3C )
        goto LABEL_12;
      goto LABEL_8;
    }
    goto LABEL_10;
  }
  sub_854AB0();
  v34[0] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077C4 != 2 && unk_4F07778 > 202310 )
    goto LABEL_4;
  v12 = (_QWORD *)sub_727060();
LABEL_11:
  v13 = dword_4F04C3C;
  if ( !dword_4F04C3C )
  {
LABEL_8:
    a2 = 69;
    a1 = (__int64)v12;
    sub_8699D0(v12, 69, 0);
  }
LABEL_12:
  sub_7B8B50(a1, a2, v13, v11);
  v16 = qword_4F061C8;
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  ++*(_BYTE *)(v16 + 36);
  ++*(_BYTE *)(v16 + 75);
  sub_7BE280(27, 125, 0, 0);
  sub_6B9B50(v33);
  --*(_BYTE *)(qword_4F061C8 + 75LL);
  if ( word_4F06418[0] == 28 )
  {
    if ( !dword_4F07760 )
      sub_684AA0(7, 2783, &dword_4F063F8);
    sub_7BE280(28, 18, 0, 0);
    v17 = v33;
    v18 = 0;
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    v19 = *(_BYTE *)(v17 + 173);
    if ( !v19 )
      goto LABEL_16;
LABEL_33:
    if ( v19 == 12 || !(unsigned int)sub_711520() )
    {
      if ( v12 )
      {
        *v12 = sub_73A460(v33);
        if ( v18 )
          v12[1] = sub_73A460(v18);
        v12[2] = v34[0];
      }
    }
    else if ( v18 )
    {
      v25 = v18[22];
      v32 = qword_4F06B40[v18[21] & 7];
      v31 = v25 / v32;
      if ( v31 + 1 > unk_4F06C48 )
      {
        v30 = v18[22];
        sub_729510(v25 / v32 + 1, v32, v25 % v32);
        v25 = v30;
      }
      v26 = v18[23];
      v27 = 0;
      if ( v25 >= v32 )
      {
        do
        {
          v28 = sub_722AB0(v26, v32);
          if ( !v28 )
            break;
          if ( v28 > 0x7F || (v29 = v28, (unsigned int)sub_7B3970((unsigned int)v28)) )
            *((_BYTE *)qword_4F06C50 + v27) = 63;
          else
            *((_BYTE *)qword_4F06C50 + v27) = v29;
          ++v27;
          v26 += v32;
        }
        while ( v31 > v27 );
      }
      *((_BYTE *)qword_4F06C50 + v27) = 0;
      sub_6851A0(1574, v34, qword_4F06C50);
    }
    else
    {
      sub_6851C0(2784, v34);
    }
    goto LABEL_16;
  }
  sub_7BE280(67, 253, 0, 0);
  if ( word_4F06418[0] != 7 )
  {
    sub_6851D0(1038);
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    if ( v6 )
      goto LABEL_17;
LABEL_20:
    sub_7BE280(75, 65, 0, 0);
    goto LABEL_17;
  }
  sub_7B8B50(67, 253, v21, v22);
  sub_7BE280(28, 18, 0, 0);
  v24 = v33;
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  v19 = *(_BYTE *)(v24 + 173);
  if ( v19 && unk_4F063AD )
  {
    v18 = &unk_4F06300;
    goto LABEL_33;
  }
LABEL_16:
  if ( !v6 )
    goto LABEL_20;
LABEL_17:
  --*(_BYTE *)(qword_4F061C8 + 83LL);
  return sub_724E30(&v33);
}
