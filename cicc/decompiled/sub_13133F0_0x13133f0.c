// Function: sub_13133F0
// Address: 0x13133f0
//
void __fastcall sub_13133F0(__int64 a1, __int64 a2)
{
  _BOOL4 v2; // r9d
  __int64 v4; // rdx
  __int64 *v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // r8
  char v8; // bl
  unsigned __int64 v9; // r10
  char v10; // r13
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  int *v14; // rcx
  _QWORD *v15; // rsi
  int *v16; // rax
  int *v17; // r13
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  int *v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // [rsp+0h] [rbp-60h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  unsigned __int64 v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  unsigned __int64 v37; // [rsp+18h] [rbp-48h]
  _BOOL4 v38; // [rsp+20h] [rbp-40h]
  unsigned __int64 v39; // [rsp+20h] [rbp-40h]
  _BOOL4 v40; // [rsp+20h] [rbp-40h]
  unsigned __int64 v41; // [rsp+20h] [rbp-40h]
  char v42; // [rsp+2Ch] [rbp-34h]
  _BOOL4 v43; // [rsp+2Ch] [rbp-34h]

  v2 = 0;
  v4 = **(_QWORD **)(a2 + 8);
  v5 = *(__int64 **)(a2 + 16);
  v6 = *v5;
  *v5 = v4;
  v7 = v4 - v6;
  if ( *(_BYTE *)(a1 + 816) <= 2u )
    v2 = *(_BYTE *)(a1 + 1) == 0;
  v8 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 )
  {
    if ( unk_4C6F1F0 )
    {
      v23 = *(_QWORD *)(a1 + 40);
      if ( v7 < v23 )
      {
        v42 = 0;
        v9 = v6 + v23 - v4;
      }
      else
      {
        v31 = v6;
        v34 = v4;
        v37 = v4 - v6;
        v40 = v2;
        if ( v2 )
        {
          v28 = sub_1310060();
          v6 = v31;
          v4 = v34;
          v42 = v8;
          v7 = v37;
          v2 = v40;
          v9 = v28;
        }
        else
        {
          v24 = sub_1310070();
          v42 = 0;
          v2 = 0;
          v7 = v37;
          v4 = v34;
          v9 = v24;
          v6 = v31;
        }
      }
      *(_QWORD *)(a1 + 40) = v9;
    }
    else
    {
      v42 = 0;
      v9 = -1;
    }
    v10 = 0;
    if ( qword_4C6F130[0] >= 0LL )
    {
      v11 = *(_QWORD *)(a1 + 72);
      if ( v7 < v11 )
      {
        v10 = 0;
        v12 = v6 + v11 - v4;
      }
      else
      {
        v29 = v9;
        v30 = v6;
        v32 = v4;
        v35 = v7;
        v38 = v2;
        if ( v2 )
        {
          v12 = sub_130F8B0(a1);
          v10 = v8;
          v9 = v29;
          v6 = v30;
          v4 = v32;
          v7 = v35;
          v2 = v38;
        }
        else
        {
          v12 = sub_130F8C0(a1);
          v2 = 0;
          v7 = v35;
          v10 = 0;
          v4 = v32;
          v6 = v30;
          v9 = v29;
        }
      }
      *(_QWORD *)(a1 + 72) = v12;
      if ( v9 > v12 )
        v9 = v12;
    }
    v13 = *(_QWORD *)(a1 + 88);
    if ( v7 >= v13 )
    {
      v41 = v9;
      if ( v2 )
      {
        v25 = sub_134B2D0(a1);
      }
      else
      {
        v25 = sub_134B2E0(a1);
        v8 = 0;
      }
      v9 = v41;
      v14 = (int *)v25;
    }
    else
    {
      v8 = 0;
      v14 = (int *)(v6 + v13 - v4);
    }
    *(_QWORD *)(a1 + 88) = v14;
    v15 = *(_QWORD **)(a2 + 16);
    v16 = &dword_400000;
    if ( v9 <= (unsigned __int64)&dword_400000 )
      v16 = (int *)v9;
    if ( v16 > v14 )
      v16 = v14;
    **(_QWORD **)(a2 + 24) = (char *)v16 + *v15;
    sub_1313270(a1);
    if ( unk_4C6F1F0 && v42 )
      sub_13114C0(a1);
    if ( qword_4C6F130[0] >= 0LL && v10 )
    {
      v26 = *(_QWORD *)(a1 + 8);
      v27 = *(_QWORD *)(a1 + 80);
      *(_QWORD *)(a1 + 80) = v26;
      sub_130F8D0(a1, v26 - v27);
    }
    if ( v8 )
      sub_134B2F0(a1, -1);
    return;
  }
  v17 = &dword_400000;
  if ( unk_4C6F1F0 )
  {
    v21 = *(_QWORD *)(a1 + 48);
    if ( v7 < v21 )
    {
      v22 = (int *)(v6 + v21 - v4);
    }
    else
    {
      v33 = v6;
      v36 = v4;
      v39 = v4 - v6;
      v43 = v2;
      if ( v2 )
      {
        v22 = (int *)sub_1310080();
        v8 = 1;
        v6 = v33;
        v4 = v36;
        v7 = v39;
        v2 = v43;
      }
      else
      {
        v22 = (int *)sub_1310090();
        v2 = 0;
        v7 = v39;
        v4 = v36;
        v6 = v33;
      }
    }
    v17 = &dword_400000;
    *(_QWORD *)(a1 + 48) = v22;
    if ( v22 <= &dword_400000 )
      v17 = v22;
    v18 = *(_QWORD *)(a1 + 96);
    if ( v7 < v18 )
      goto LABEL_28;
  }
  else
  {
    v18 = *(_QWORD *)(a1 + 96);
    if ( v7 < v18 )
    {
LABEL_28:
      v19 = v18 + v6 - v4;
LABEL_29:
      *(_QWORD *)(a1 + 96) = v19;
      if ( (unsigned __int64)v17 <= v19 )
        v19 = (unsigned __int64)v17;
      **(_QWORD **)(a2 + 24) = **(_QWORD **)(a2 + 16) + v19;
      sub_1313270(a1);
      if ( unk_4C6F1F0 && v8 )
        sub_13114D0(a1);
      return;
    }
  }
  if ( !v2 )
  {
    v19 = sub_134B350(a1);
    goto LABEL_29;
  }
  v20 = sub_134B340(a1);
  *(_QWORD *)(a1 + 96) = v20;
  if ( v20 > (unsigned __int64)v17 )
    v20 = (unsigned __int64)v17;
  **(_QWORD **)(a2 + 24) = **(_QWORD **)(a2 + 16) + v20;
  sub_1313270(a1);
  if ( unk_4C6F1F0 && v8 )
    sub_13114D0(a1);
  sub_134B360(a1, -1);
}
