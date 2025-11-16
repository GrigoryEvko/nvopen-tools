// Function: sub_30D2590
// Address: 0x30d2590
//
unsigned __int64 __fastcall sub_30D2590(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  unsigned __int64 result; // rax
  __int64 v8; // r13
  __int64 v9; // rdx
  int v10; // eax
  int v11; // ebx
  int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rdx
  int v18; // eax
  __int64 (__fastcall *v19)(_QWORD, __int64); // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r13
  char v24; // al
  unsigned __int64 v25; // r13
  __int64 v26; // rsi
  __int64 v27; // rdx
  int v28; // eax
  int v29; // esi
  __int64 v30; // rdx
  __int64 v31; // rdi
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  int v36; // eax
  unsigned __int64 v37; // [rsp+0h] [rbp-80h]
  int v38; // [rsp+8h] [rbp-78h]
  char v39; // [rsp+Fh] [rbp-71h]
  __int64 *v40; // [rsp+10h] [rbp-70h]
  unsigned int v41; // [rsp+18h] [rbp-68h]
  int v42; // [rsp+1Ch] [rbp-64h]
  __int64 v43; // [rsp+28h] [rbp-58h] BYREF
  unsigned __int64 v44[2]; // [rsp+30h] [rbp-50h] BYREF
  char v45; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)a2 == 34 )
  {
    v6 = *(_QWORD *)(a2 - 96);
    result = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( result == v6 + 48 )
      goto LABEL_79;
    if ( !result )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(result - 24) - 30 > 0xA )
LABEL_79:
      BUG();
  }
  else
  {
    v14 = *(_QWORD *)(a2 + 40);
    result = *(_QWORD *)(v14 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( result == v14 + 48 )
      goto LABEL_80;
    if ( !result )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(result - 24) - 30 > 0xA )
LABEL_80:
      BUG();
  }
  if ( *(_BYTE *)(result - 24) == 36 )
  {
    *(_DWORD *)(a1 + 704) = 0;
    return result;
  }
  v8 = sub_B491C0(a2);
  v42 = sub_DF94D0(*(_QWORD *)(a1 + 8));
  v41 = sub_DF9440(*(_QWORD *)(a1 + 8));
  if ( (unsigned __int8)sub_B2D610(v8, 18) )
  {
    v9 = *(_QWORD *)(a1 + 664);
    v10 = *(_DWORD *)(a1 + 704);
    if ( *(_BYTE *)(v9 + 32) && v10 > *(_DWORD *)(v9 + 28) )
      v10 = *(_DWORD *)(v9 + 28);
    *(_DWORD *)(a1 + 704) = v10;
    v11 = 0;
    v42 = 0;
  }
  else
  {
    if ( (unsigned __int8)sub_B2D610(v8, 47) || (unsigned __int8)sub_B2D610(v8, 18) )
    {
      v15 = *(_QWORD *)(a1 + 664);
      v16 = *(_DWORD *)(a1 + 704);
      if ( *(_BYTE *)(v15 + 24) && v16 > *(_DWORD *)(v15 + 20) )
        v16 = *(_DWORD *)(v15 + 20);
      *(_DWORD *)(a1 + 704) = v16;
    }
    v11 = 50;
  }
  if ( (unsigned __int8)sub_B2D610(v8, 18) )
    goto LABEL_10;
  if ( (unsigned __int8)sub_B2D610(a3, 16) )
  {
    v17 = *(_QWORD *)(a1 + 664);
    v18 = *(_DWORD *)(a1 + 704);
    if ( *(_BYTE *)(v17 + 8) && v18 < *(_DWORD *)(v17 + 4) )
      v18 = *(_DWORD *)(v17 + 4);
    *(_DWORD *)(a1 + 704) = v18;
  }
  v19 = *(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 32);
  if ( !v19 )
  {
    v31 = *(_QWORD *)(a1 + 64);
    if ( v31 )
    {
      if ( *(_QWORD *)(v31 + 8) )
      {
        v40 = 0;
        if ( sub_D84510(v31, a2, 0) )
          goto LABEL_34;
      }
    }
LABEL_70:
    v40 = 0;
    goto LABEL_37;
  }
  v20 = v19(*(_QWORD *)(a1 + 40), v8);
  v21 = *(_QWORD *)(a1 + 64);
  v40 = (__int64 *)v20;
  if ( v21 && *(_QWORD *)(v21 + 8) && sub_D84510(v21, a2, v20) )
  {
LABEL_34:
    v22 = *(_QWORD *)(a1 + 664);
    v38 = *(_DWORD *)(v22 + 36);
    v39 = *(_BYTE *)(v22 + 40);
    goto LABEL_38;
  }
  if ( !v40 )
    goto LABEL_70;
  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 664) + 48LL) )
  {
LABEL_37:
    v39 = 0;
    goto LABEL_38;
  }
  v37 = sub_FDD860(v40, *(_QWORD *)(a2 + 40));
  v43 = sub_FDC4B0((__int64)v40);
  v32 = sub_1098D90((unsigned __int64 *)&v43, qword_5030248);
  v39 = 0;
  v44[1] = v33;
  v44[0] = v32;
  if ( (_BYTE)v33 && v44[0] <= v37 )
  {
    v34 = *(_QWORD *)(a1 + 664);
    v38 = *(_DWORD *)(v34 + 44);
    v39 = *(_BYTE *)(v34 + 48);
  }
LABEL_38:
  if ( !(unsigned __int8)sub_B2D610(v8, 47) && !(unsigned __int8)sub_B2D610(v8, 18) && v39 )
  {
    *(_DWORD *)(a1 + 704) = v38;
    goto LABEL_10;
  }
  v23 = *(_QWORD *)(a1 + 64);
  if ( v23 && *(_QWORD *)(v23 + 8) )
  {
    v24 = sub_D84560(*(_QWORD *)(a1 + 64), a2, (__int64)v40);
  }
  else
  {
    if ( !v40 )
      goto LABEL_52;
    sub_F02DB0(&v43, dword_5030328, 0x64u);
    v25 = sub_FDD860(v40, *(_QWORD *)(a2 + 40));
    v26 = *(_QWORD *)(sub_B491C0(a2) + 80);
    if ( v26 )
      v26 -= 24;
    v44[0] = sub_FDD860(v40, v26);
    v24 = sub_1098D20(v44, v43) > v25;
  }
  if ( v24 )
  {
    v27 = *(_QWORD *)(a1 + 664);
    v28 = *(_DWORD *)(a1 + 704);
    v29 = *(_DWORD *)(v27 + 52);
    if ( *(_BYTE *)(v27 + 56) )
      goto LABEL_48;
    goto LABEL_50;
  }
  v23 = *(_QWORD *)(a1 + 64);
LABEL_52:
  if ( v23 )
  {
    if ( *(_QWORD *)(v23 + 8) )
    {
      sub_B2EE70((__int64)v44, a3, 0);
      if ( v45 && sub_D84440(v23, v44[0]) )
      {
        v35 = *(_QWORD *)(a1 + 664);
        v36 = *(_DWORD *)(a1 + 704);
        if ( *(_BYTE *)(v35 + 8) && v36 < *(_DWORD *)(v35 + 4) )
          v36 = *(_DWORD *)(v35 + 4);
        *(_DWORD *)(a1 + 704) = v36;
        goto LABEL_10;
      }
      v23 = *(_QWORD *)(a1 + 64);
    }
    if ( sub_D84460(v23, a3) )
    {
      v30 = *(_QWORD *)(a1 + 664);
      v28 = *(_DWORD *)(a1 + 704);
      v29 = *(_DWORD *)(v30 + 12);
      if ( *(_BYTE *)(v30 + 16) )
      {
LABEL_48:
        if ( v28 > v29 )
          v28 = v29;
      }
LABEL_50:
      *(_DWORD *)(a1 + 704) = v28;
      v11 = 0;
      v41 = 0;
      v42 = 0;
    }
  }
LABEL_10:
  *(_DWORD *)(a1 + 704) += sub_DF9470(*(_QWORD *)(a1 + 8));
  v12 = *(_DWORD *)(a1 + 704) * sub_DF93B0(*(_QWORD *)(a1 + 8));
  v13 = *(_QWORD *)(a1 + 72);
  *(_DWORD *)(a1 + 704) = v12;
  *(_DWORD *)(a1 + 660) = v12 * v11 / 100;
  *(_DWORD *)(a1 + 656) = v42 * v12 / 100;
  result = sub_30D14D0(a2, v13);
  if ( (_BYTE)result )
  {
    *(_DWORD *)(a1 + 716) -= v41;
    *(_DWORD *)(a1 + 708) = v41;
    return v41;
  }
  return result;
}
