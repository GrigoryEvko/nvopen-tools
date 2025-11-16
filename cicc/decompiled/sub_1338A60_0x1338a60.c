// Function: sub_1338A60
// Address: 0x1338a60
//
__int64 __fastcall sub_1338A60(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        unsigned int *a4,
        __int64 *a5,
        unsigned int *a6,
        __int64 a7)
{
  __int64 v9; // r13
  unsigned int v10; // eax
  unsigned int v11; // ecx
  unsigned int v12; // eax
  unsigned int v13; // r15d
  unsigned int v14; // r9d
  unsigned int v16; // eax
  unsigned int v17; // eax
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // eax
  __int64 v23; // rcx
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *v27; // [rsp+8h] [rbp-48h]
  __int64 *v28; // [rsp+8h] [rbp-48h]
  __int64 *v29; // [rsp+8h] [rbp-48h]
  __int64 *v30; // [rsp+8h] [rbp-48h]
  _DWORD v31[13]; // [rsp+1Ch] [rbp-34h]

  if ( *(char *)(a1 + 1) > 0 )
  {
    v9 = qword_50579C0[0];
    if ( qword_50579C0[0] )
      goto LABEL_20;
    a2 = 0;
    v30 = a5;
    v25 = sub_1300B80(a1, 0, (__int64)&off_49E8000);
    a5 = v30;
    v9 = v25;
    goto LABEL_42;
  }
  v9 = *(_QWORD *)(a1 + 144);
  if ( !v9 )
  {
    a2 = 0;
    v28 = a5;
    v19 = sub_1302AE0(a1, 0);
    a5 = v28;
    v9 = v19;
    if ( *(_BYTE *)a1 )
    {
      v20 = *(_QWORD *)(a1 + 296);
      a2 = (_QWORD *)(a1 + 256);
      v21 = a1 + 856;
      if ( v20 )
      {
        if ( v9 == v20 )
        {
          a3 = unk_4C6F238;
          if ( unk_4C6F238 > 2u )
            goto LABEL_4;
LABEL_20:
          v10 = *(_DWORD *)(v9 + 78928);
          goto LABEL_21;
        }
        sub_1311F50(a1, a2, v21, v9);
        a5 = v28;
      }
      else
      {
        sub_13114E0(a1, a2, v21, v9);
        a5 = v28;
      }
    }
    a3 = unk_4C6F238;
    if ( unk_4C6F238 > 2u )
      goto LABEL_4;
LABEL_42:
    if ( !v9 )
      return 11;
    goto LABEL_20;
  }
  a3 = unk_4C6F238;
  if ( unk_4C6F238 <= 2u )
    goto LABEL_20;
LABEL_4:
  a2 = &dword_505F9BC;
  v10 = *(_DWORD *)(v9 + 78928);
  v11 = dword_505F9BC;
  if ( (_DWORD)a3 == 4 && dword_505F9BC > 1u )
  {
    a3 = dword_505F9BC & 1;
    v11 = (dword_505F9BC >> 1) - (((_DWORD)a3 == 0) - 1);
  }
  if ( v10 >= v11 || a1 == *(_QWORD *)(v9 + 16) )
  {
LABEL_21:
    v31[0] = v10;
    if ( !a6 )
      goto LABEL_15;
LABEL_22:
    v14 = 22;
    if ( a7 != 4 )
      return v14;
    v13 = *a6;
    if ( !a4 || !a5 )
      goto LABEL_27;
    goto LABEL_25;
  }
  v27 = a5;
  v12 = sched_getcpu();
  a5 = v27;
  if ( unk_4C6F238 != 3 )
  {
    a3 = dword_505F9BC >> 1;
    a2 = (_QWORD *)(v12 - (unsigned int)a3);
    if ( (unsigned int)a3 <= v12 )
      v12 -= a3;
  }
  if ( *(_DWORD *)(v9 + 78928) != v12 )
  {
    v9 = *(_QWORD *)(a1 + 144);
    if ( v12 != *(_DWORD *)(v9 + 78928) )
    {
      v24 = qword_50579C0[v12];
      if ( !v24 )
      {
        v26 = sub_1300B80(a1, v12, (__int64)&off_49E8000);
        a5 = v27;
        v24 = v26;
      }
      a2 = (_QWORD *)v9;
      v29 = a5;
      sub_1302A70(a1, v9, v24);
      a5 = v29;
      if ( *(_BYTE *)a1 )
      {
        a2 = (_QWORD *)(a1 + 256);
        sub_1311F50(a1, (_QWORD *)(a1 + 256), a1 + 856, v24);
        v9 = *(_QWORD *)(a1 + 144);
        a5 = v29;
      }
      else
      {
        v9 = *(_QWORD *)(a1 + 144);
      }
    }
  }
  v10 = *(_DWORD *)(v9 + 78928);
  *(_QWORD *)(v9 + 16) = a1;
  v31[0] = v10;
  if ( a6 )
    goto LABEL_22;
LABEL_15:
  if ( !a4 )
    return 0;
  v13 = v10;
  if ( !a5 )
    return 0;
LABEL_25:
  a3 = *a5;
  if ( *a5 != 4 )
  {
    if ( (unsigned __int64)*a5 > 4 )
      a3 = 4;
    if ( (_DWORD)a3 )
    {
      v22 = 0;
      do
      {
        v23 = v22++;
        *((_BYTE *)a4 + v23) = *((_BYTE *)v31 + v23);
      }
      while ( v22 < (unsigned int)a3 );
    }
    *a5 = a3;
    return 22;
  }
  *a4 = v10;
LABEL_27:
  if ( v10 == v13 )
    return 0;
  v16 = sub_1300B70(a1, a2, a3);
  v14 = 14;
  if ( v16 > v13 )
  {
    if ( unk_4C6F238 <= 2u )
      goto LABEL_34;
    v17 = dword_505F9BC;
    if ( unk_4C6F238 == 4 && dword_505F9BC > 1u )
      v17 = (dword_505F9BC >> 1) - (((dword_505F9BC & 1) == 0) - 1);
    v14 = 1;
    if ( v17 <= v13 )
    {
LABEL_34:
      v18 = qword_50579C0[v13];
      if ( v18 || (v18 = sub_1300B80(a1, v13, (__int64)&off_49E8000)) != 0 )
      {
        sub_1302A70(a1, v9, v18);
        if ( *(_BYTE *)a1 )
        {
          sub_1311F50(a1, (_QWORD *)(a1 + 256), a1 + 856, v18);
          return 0;
        }
        return 0;
      }
      return 11;
    }
  }
  return v14;
}
