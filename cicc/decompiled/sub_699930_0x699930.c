// Function: sub_699930
// Address: 0x699930
//
__int64 __fastcall sub_699930(__int64 a1, __m128i *a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r14
  char v8; // dl
  __int64 v9; // rax
  int v10; // r15d
  __int64 v11; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdi
  __int64 v20; // rsi
  int v21; // eax
  __m128i **v22; // r9
  unsigned __int8 *v23; // rax
  unsigned __int8 *v24; // rax
  int v25; // r14d
  int v26; // [rsp+28h] [rbp-98h]
  __int64 *v27; // [rsp+30h] [rbp-90h]
  __int64 v29; // [rsp+40h] [rbp-80h] BYREF
  __int64 v30; // [rsp+48h] [rbp-78h] BYREF
  __m128i v31[7]; // [rsp+50h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a1 + 80);
  sub_6E1DD0(&v29);
  v7 = *(_QWORD *)(*(_QWORD *)(v6 + 16) + 120LL);
  if ( (unsigned int)sub_8D32E0(v7) )
  {
    v7 = sub_8D46C0(v7);
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
  }
  else if ( dword_4F077C4 != 2 )
  {
    goto LABEL_3;
  }
  if ( (unsigned int)sub_8D23B0(v7) )
    sub_8AE000(v7);
LABEL_3:
  v8 = *(_BYTE *)(v7 + 140);
  if ( v8 == 12 )
  {
    v9 = v7;
    do
    {
      v9 = *(_QWORD *)(v9 + 160);
      v8 = *(_BYTE *)(v9 + 140);
    }
    while ( v8 == 12 );
  }
  if ( !v8 )
    goto LABEL_7;
  v10 = sub_8DBE70(v7);
  if ( v10 )
  {
    v10 = 1;
    goto LABEL_8;
  }
  if ( (unsigned int)sub_8D3410(v7) )
  {
    if ( !(unsigned int)sub_68C2C0(a2, (__int64 *)v6, v15, v16, v17, v18) )
    {
LABEL_7:
      v10 = 0;
      goto LABEL_8;
    }
  }
  else
  {
    v21 = sub_8D3A70(v7);
    v22 = (__m128i **)(v6 + 40);
    v27 = (__int64 *)(v6 + 48);
    if ( v21
      && (v23 = sub_694FD0(v7, "begin", v31), v22 = (__m128i **)(v6 + 40), v23)
      && (v24 = sub_694FD0(v7, "end", v31), v22 = (__m128i **)(v6 + 40), v24) )
    {
      v25 = sub_698D30(
              *(_QWORD *)(v6 + 16),
              (__m128i *)"begin",
              a2,
              a3,
              *(_BYTE *)(v6 + 72) & 1,
              (__int64 *)(v6 + 40),
              &v30);
      if ( !(unsigned int)sub_698D30(*(_QWORD *)(v6 + 16), (__m128i *)"end", a2, a3, 0, v27, v31) )
        goto LABEL_8;
      if ( !v25 )
        goto LABEL_7;
    }
    else
    {
      v26 = sub_698BA0(*(_QWORD *)(v6 + 16), (__m128i *)"begin", (__int64)a2, a3, *(_BYTE *)(v6 + 72) & 1, v22);
      if ( !(unsigned int)sub_698BA0(*(_QWORD *)(v6 + 16), (__m128i *)"end", (__int64)a2, a3 + 1, 0, (__m128i **)v27)
        || !v26 )
      {
        goto LABEL_7;
      }
    }
  }
  if ( !unk_4F07724 )
  {
    v19 = *(_QWORD *)(*(_QWORD *)(v6 + 40) + 120LL);
    v20 = *(_QWORD *)(*(_QWORD *)(v6 + 48) + 120LL);
    if ( v19 != v20 && !(unsigned int)sub_8DED30(v19, v20, 1) )
    {
      sub_6861A0(0x8F0u, a2, *(_QWORD *)(*(_QWORD *)(v6 + 40) + 120LL), *(_QWORD *)(*(_QWORD *)(v6 + 48) + 120LL));
      goto LABEL_7;
    }
  }
  if ( !*(_QWORD *)(v6 + 8) )
    return sub_6E1DF0(v29);
  sub_698F70(v6, a2, a3, a4);
LABEL_8:
  v11 = *(_QWORD *)(v6 + 8);
  if ( v11 )
  {
    if ( (*(_BYTE *)(v11 + 175) & 7) != 0 )
    {
      v13 = sub_8D4940(*(_QWORD *)(v11 + 120));
      if ( (unsigned int)sub_8D3EA0(v13) )
      {
        if ( v10 )
          v14 = *(_QWORD *)&dword_4D03B80;
        else
          v14 = sub_72C930(v13);
        *(_QWORD *)(*(_QWORD *)(v6 + 8) + 120LL) = v14;
      }
      v11 = *(_QWORD *)(v6 + 8);
    }
    sub_8756B0(*(_QWORD *)v11);
    sub_8767A0(4, **(_QWORD **)(v6 + 8), *(_QWORD *)(v6 + 8) + 64LL, 1);
  }
  return sub_6E1DF0(v29);
}
