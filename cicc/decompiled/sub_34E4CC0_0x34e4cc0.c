// Function: sub_34E4CC0
// Address: 0x34e4cc0
//
__int64 __fastcall sub_34E4CC0(__int64 a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // rsi
  int v6; // r13d
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // r13d
  unsigned __int64 v12; // rax
  unsigned __int8 v13; // dl
  __int64 v14; // rax
  __int64 v15; // rcx
  int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // rsi
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-30h]
  unsigned __int64 v23; // [rsp+8h] [rbp-28h] BYREF

  if ( qword_503AFB0 | qword_503B0B0 )
  {
    sub_DFE400(a2);
    v4 = sub_34E26C0(qword_503B0A8, qword_503B0B0);
    v5 = qword_503AFB0;
    v6 = v4;
    LODWORD(v7) = sub_34E26C0(qword_503AFA8, qword_503AFB0);
  }
  else
  {
    v5 = a1;
    v12 = sub_DFE400(a2);
    v6 = v12;
    v7 = HIDWORD(v12);
  }
  v8 = *(_QWORD *)(a1 - 32);
  if ( !v8 || *(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  if ( sub_B5B000(*(_DWORD *)(v8 + 36)) )
  {
LABEL_7:
    if ( v6 == 1 || (_DWORD)v7 == 2 )
    {
LABEL_15:
      sub_34E30B0(a1);
      v10 = v13;
      goto LABEL_11;
    }
LABEL_9:
    if ( v6 != 2 )
    {
LABEL_10:
      v10 = 0;
      goto LABEL_11;
    }
    goto LABEL_15;
  }
  v14 = *(_QWORD *)(a1 - 32);
  if ( !v14 || *(_BYTE *)v14 || (v15 = *(_QWORD *)(a1 + 80), *(_QWORD *)(v14 + 24) != v15) )
    BUG();
  v22 = sub_B5A9F0(*(_DWORD *)(v14 + 36), v5, v9, v15);
  if ( BYTE4(v22) )
  {
    v17 = (__int64 *)sub_BD5C60(a1);
    v18 = 67;
    v23 = sub_B612D0(v17, v22);
    v19 = sub_A73ED0(&v23, 67);
  }
  else
  {
    v20 = *(_QWORD *)(a1 - 32);
    if ( !v20 || *(_BYTE *)v20 || (v21 = *(_QWORD *)(a1 + 80), *(_QWORD *)(v20 + 24) != v21) )
      BUG();
    v23 = sub_B5A790(*(_DWORD *)(v20 + 36), v5, v21, v16);
    if ( !BYTE4(v23) )
      goto LABEL_7;
    v18 = a1;
    v19 = sub_991600(v23, a1, 0, 0, 0, 0, 1, 0);
  }
  if ( !v19 )
    goto LABEL_7;
  if ( (_DWORD)v7 != 2 && v6 != 1 )
    goto LABEL_9;
  if ( sub_B5AF00(a1, v18) || !sub_B5A450(a1) )
    goto LABEL_10;
  v10 = (unsigned __int8)sub_34E2BC0(a1);
LABEL_11:
  if ( (_DWORD)v7 == 1 )
    BUG();
  if ( (_DWORD)v7 != 2 )
    return v10;
  if ( a1 != sub_34E3680(a1) )
    return 2;
  return v10;
}
