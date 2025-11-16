// Function: sub_303B880
// Address: 0x303b880
//
__int64 __fastcall sub_303B880(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  int v5; // r14d
  __int64 v7; // rbx
  unsigned __int16 *v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // dl
  unsigned __int64 v13; // rsi
  __int64 *v14; // rcx
  __int16 v15; // cx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r15
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int128 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 *v25; // [rsp+0h] [rbp-70h]
  __int16 v26; // [rsp+0h] [rbp-70h]
  __int128 v27; // [rsp+0h] [rbp-70h]
  unsigned int v28; // [rsp+20h] [rbp-50h] BYREF
  __int64 v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h] BYREF
  int v31; // [rsp+38h] [rbp-38h]

  v7 = a2;
  if ( !sub_3037D80(a1, *(__int64 **)(a4 + 40)) )
    return v7;
  v9 = *(unsigned __int16 **)(a2 + 48);
  LODWORD(v10) = *v9;
  v11 = *((_QWORD *)v9 + 1);
  LOWORD(v28) = v10;
  v29 = v11;
  if ( (_WORD)v10 )
  {
    if ( (unsigned __int16)(v10 - 17) <= 0xD3u )
    {
      v12 = (unsigned __int16)(v10 - 176) <= 0x34u;
      LODWORD(v13) = word_4456340[(int)v10 - 1];
      LOBYTE(v10) = v12;
LABEL_6:
      v14 = *(__int64 **)(a4 + 64);
      LODWORD(v30) = v13;
      BYTE4(v30) = v10;
      v25 = v14;
      if ( v12 )
        v15 = sub_2D43AD0(12, v13);
      else
        v15 = sub_2D43050(12, v13);
      v18 = 0;
      if ( !v15 )
      {
        v5 = sub_3009450(v25, 12, 0, v30, v16, v17);
        v15 = v5;
        v18 = v24;
      }
      HIWORD(v4) = HIWORD(v5);
      goto LABEL_13;
    }
  }
  else if ( sub_30070B0((__int64)&v28) )
  {
    v13 = sub_3007240((__int64)&v28);
    v10 = HIDWORD(v13);
    v12 = BYTE4(v13);
    goto LABEL_6;
  }
  v18 = 0;
  v15 = 12;
LABEL_13:
  v19 = *(_QWORD *)(v7 + 80);
  v30 = v19;
  if ( v19 )
  {
    v26 = v15;
    sub_B96E90((__int64)&v30, v19, 1);
    v15 = v26;
  }
  LOWORD(v4) = v15;
  v31 = *(_DWORD *)(v7 + 72);
  *(_QWORD *)&v27 = sub_3406EE0(a4, **(_QWORD **)(v7 + 40), *(_QWORD *)(*(_QWORD *)(v7 + 40) + 8LL), &v30, v4, v18);
  *((_QWORD *)&v27 + 1) = v20;
  *(_QWORD *)&v21 = sub_3406EE0(
                      a4,
                      *(_QWORD *)(*(_QWORD *)(v7 + 40) + 40LL),
                      *(_QWORD *)(*(_QWORD *)(v7 + 40) + 48LL),
                      &v30,
                      v4,
                      v18);
  v22 = sub_3405C90(a4, *(_DWORD *)(v7 + 24), (unsigned int)&v30, v4, v18, *(_DWORD *)(v7 + 28), v27, v21);
  v7 = sub_3406EE0(a4, v22, v23, &v30, v28, v29);
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v7;
}
