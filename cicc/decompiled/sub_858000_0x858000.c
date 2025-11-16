// Function: sub_858000
// Address: 0x858000
//
void __fastcall sub_858000(unsigned __int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // r14
  char *v11; // rax
  __int64 v12; // rsi
  unsigned __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  char *v19; // rax
  unsigned __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  char v25; // r13
  char *v26; // rax
  unsigned int *v27; // rsi
  unsigned __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9

  sub_7C9660(a1);
  if ( word_4F06418[0] != 27 )
  {
    if ( word_4F06418[0] != 7 )
    {
LABEL_3:
      sub_7C96B0(0, (unsigned int *)a2, v2, v3, v4, v5);
      return;
    }
    if ( !unk_4F063AD )
    {
      sub_7B8B50(a1, (unsigned int *)a2, v2, v3, v4, v5);
      goto LABEL_3;
    }
    v6 = 0;
    v7 = qword_4F04C50;
    if ( !qword_4F04C50 )
    {
      v10 = *(_QWORD *)word_4F063B0;
      v19 = (char *)sub_822B10(*(__int64 *)word_4F063B0, a2, v2, v3, v4, v5);
      v12 = (__int64)qword_4F063B8;
      v20 = (unsigned __int64)v19;
      v14 = (__int64)v19;
      strcpy(v19, qword_4F063B8);
      sub_7B8B50(v20, (unsigned int *)v12, v21, v22, v23, v24);
      goto LABEL_17;
    }
    goto LABEL_6;
  }
  sub_7B8B50(a1, (unsigned int *)a2, v2, v3, v4, v5);
  if ( word_4F06418[0] != 7 )
    goto LABEL_12;
  if ( !unk_4F063AD )
  {
    sub_7B8B50(a1, (unsigned int *)a2, v2, v3, v4, v5);
    goto LABEL_12;
  }
  v6 = 1;
  v7 = qword_4F04C50;
  if ( qword_4F04C50 )
  {
LABEL_6:
    v8 = *(_QWORD *)(v7 + 32);
    if ( !v8 || (v9 = *(_BYTE *)(v8 + 198), v2 = v9 & 0x18, (_BYTE)v2 == 16) || (v9 & 0x10) != 0 )
    {
      v10 = *(_QWORD *)word_4F063B0;
      v11 = (char *)sub_822B10(*(__int64 *)word_4F063B0, a2, v2, v3, v4, v5);
      v12 = (__int64)qword_4F063B8;
      v13 = (unsigned __int64)v11;
      v14 = (__int64)v11;
      strcpy(v11, qword_4F063B8);
      sub_7B8B50(v13, (unsigned int *)v12, v15, v16, v17, v18);
      if ( !v6 )
        goto LABEL_17;
      goto LABEL_22;
    }
    sub_7B8B50(a1, (unsigned int *)a2, v2, v3, v4, v5);
    if ( !v6 )
      goto LABEL_3;
LABEL_12:
    a2 = 18;
    if ( (unsigned int)sub_7BE280(0x1Cu, 18, 0, 0, v4, v5) )
      goto LABEL_3;
    goto LABEL_13;
  }
  v10 = *(_QWORD *)word_4F063B0;
  v26 = (char *)sub_822B10(*(__int64 *)word_4F063B0, a2, v2, v3, v4, v5);
  v27 = (unsigned int *)qword_4F063B8;
  v28 = (unsigned __int64)v26;
  v14 = (__int64)v26;
  strcpy(v26, qword_4F063B8);
  sub_7B8B50(v28, v27, v29, v30, v31, v32);
LABEL_22:
  v12 = 18;
  if ( !(unsigned int)sub_7BE280(0x1Cu, 18, 0, 0, v4, v5) )
  {
LABEL_13:
    sub_7C96B0(1u, (unsigned int *)0x12, v2, v3, v4, v5);
    return;
  }
LABEL_17:
  sub_7C96B0(0, (unsigned int *)v12, v2, v3, v4, v5);
  v25 = byte_4F07481[0];
  byte_4F07481[0] = 4;
  sub_684AE0(0xE49u, &dword_4F063F8, v14);
  byte_4F07481[0] = v25;
  sub_822B90(v14, v10);
}
