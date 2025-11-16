// Function: sub_23B6C50
// Address: 0x23b6c50
//
void __fastcall sub_23B6C50(char *a1, __int64 *a2, __m128i a3)
{
  void *(*v4)(); // rax
  void *v5; // rax
  __int64 v6; // r15
  __int64 v7; // rsi
  __int64 v8; // r12
  void *(*v9)(); // rax
  void *v10; // rax
  __int64 v11; // r15
  char *v12; // rax
  __int64 v13; // rdx
  void *(*v14)(); // rax
  void *v15; // rax
  __int64 v16; // r15
  char *v17; // rax
  __int64 v18; // rdx
  void *(*v19)(); // rax
  void *v20; // rax
  __int64 v21; // r15
  __int64 v22; // r12
  __int64 i; // r15
  __int64 v24; // r14
  char *v25; // rax
  __int64 v26; // rdx
  __int64 *v27; // rax
  __int64 v28; // r12
  _BYTE *v29; // rax
  __int64 v30; // rdx
  __int64 v31[8]; // [rsp+0h] [rbp-40h] BYREF

  if ( (unsigned __int8)sub_BC5DE0() )
  {
    sub_23B2720(v31, a2);
    v8 = sub_23B66B0(v31, 0);
    sub_23B42E0(v31);
    v7 = v8;
    goto LABEL_7;
  }
  sub_23B2720(v31, a2);
  if ( v31[0]
    && ((v4 = *(void *(**)())(*(_QWORD *)v31[0] + 24LL), v4 != sub_23AE340) ? (v5 = v4()) : (v5 = &unk_4CDFBF8),
        v5 == &unk_4C5D162) )
  {
    v6 = *(_QWORD *)(v31[0] + 8);
    sub_23B42E0(v31);
    v7 = v6;
    if ( v6 )
    {
LABEL_7:
      sub_23AEF60(a1, v7, a3);
      return;
    }
  }
  else
  {
    sub_23B42E0(v31);
  }
  sub_23B2720(v31, a2);
  if ( v31[0]
    && ((v9 = *(void *(**)())(*(_QWORD *)v31[0] + 24LL), v9 != sub_23AE340) ? (v10 = v9()) : (v10 = &unk_4CDFBF8),
        v10 == &unk_4C5D161) )
  {
    v11 = *(_QWORD *)(v31[0] + 8);
    sub_23B42E0(v31);
    if ( v11 )
    {
      v12 = (char *)sub_BD5D20(v11);
      if ( sub_BC63A0(v12, v13) )
        sub_A69870(v11, a1, 0);
      return;
    }
  }
  else
  {
    sub_23B42E0(v31);
  }
  sub_23B2720(v31, a2);
  if ( v31[0]
    && ((v19 = *(void *(**)())(*(_QWORD *)v31[0] + 24LL), v19 != sub_23AE340) ? (v20 = v19()) : (v20 = &unk_4CDFBF8),
        v20 == &unk_4C5D118) )
  {
    v21 = *(_QWORD *)(v31[0] + 8);
    sub_23B42E0(v31);
    if ( v21 )
    {
      v22 = *(_QWORD *)(v21 + 8);
      for ( i = v22 + 8LL * *(unsigned int *)(v21 + 16); i != v22; v22 += 8 )
      {
        v24 = *(_QWORD *)(*(_QWORD *)v22 + 8LL);
        if ( !sub_B2FC80(v24) )
        {
          v25 = (char *)sub_BD5D20(v24);
          if ( sub_BC63A0(v25, v26) )
            sub_A68C30(v24, (__int64)a1, 0, 0, 0);
        }
      }
      return;
    }
  }
  else
  {
    sub_23B42E0(v31);
  }
  sub_23B2720(v31, a2);
  if ( v31[0]
    && ((v14 = *(void *(**)())(*(_QWORD *)v31[0] + 24LL), v14 != sub_23AE340) ? (v15 = v14()) : (v15 = &unk_4CDFBF8),
        v15 == &unk_4C5D160) )
  {
    v16 = *(_QWORD *)(v31[0] + 8);
    sub_23B42E0(v31);
    if ( v16 )
    {
      v17 = (char *)sub_BD5D20(*(_QWORD *)(**(_QWORD **)(v16 + 32) + 72LL));
      if ( sub_BC63A0(v17, v18) )
      {
        sub_23B0820(v31, byte_3F871B3);
        sub_D4BD90(v16, a1, (__int64)v31, a3);
        sub_2240A30((unsigned __int64 *)v31);
      }
      return;
    }
  }
  else
  {
    sub_23B42E0(v31);
  }
  sub_23B2720(v31, a2);
  v27 = (__int64 *)sub_23B6650(v31);
  if ( !v27 )
  {
    sub_23B42E0(v31);
LABEL_48:
    BUG();
  }
  v28 = *v27;
  sub_23B42E0(v31);
  if ( !v28 )
    goto LABEL_48;
  v29 = (_BYTE *)sub_2E791E0(v28);
  if ( sub_BC63A0(v29, v30) )
    sub_2E823F0(v28, a1, 0);
}
