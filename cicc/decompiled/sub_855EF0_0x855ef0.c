// Function: sub_855EF0
// Address: 0x855ef0
//
__int64 __fastcall sub_855EF0(unsigned __int64 a1, unsigned int *a2, char **a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r12d
  const char *v10; // r14
  size_t v11; // r8
  char *v12; // rax
  size_t v13; // rdx
  unsigned __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int *v17; // rsi
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  size_t n[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = a1;
  *a2 = 0;
  if ( (unsigned __int16)sub_7B8B50(a1, a2, (__int64)a3, a4, a5, a6) == 1 )
  {
    v10 = qword_4F06410;
    v11 = qword_4F06400;
    n[0] = qword_4F06400;
    if ( unk_4F061E4 )
    {
      v28 = sub_7B3EE0((unsigned __int8 *)qword_4F06410, n);
      v11 = n[0];
      v10 = (const char *)v28;
    }
    if ( a3 )
    {
      v12 = (char *)sub_823970(v11 + 2);
      v13 = n[0];
      *a3 = v12;
      strncpy(v12, v10, v13);
      (*a3)[n[0]] = 0;
      v11 = n[0];
    }
    if ( dword_4D04788 && v11 == 11 )
    {
      if ( !memcmp(v10, "__VA_ARGS__", 0xBu) )
      {
        sub_6851C0(0x3C9u, dword_4F07508);
        v11 = n[0];
      }
    }
    else if ( unk_4D041B8 && v11 == 10 && !memcmp(v10, "__VA_OPT__", 0xAu) )
    {
      sub_6851C0(0xB7Bu, dword_4F07508);
      v11 = n[0];
    }
    v14 = sub_87A100(v10, v11, &qword_4D04A00);
    v17 = (unsigned int *)sub_81B700();
    if ( v17 )
    {
      *a2 = v7;
      v14 = 4;
      sub_8767A0(4, v17, &dword_4F063F8, 1);
    }
    else
    {
      *a2 = v7 ^ 1;
    }
    sub_7B8B50(v14, v17, v15, v16, v18, v19);
    if ( word_4F06418[0] != 10 )
      sub_855DA0(v14, (__int64)v17, v20, v21, v22, v23);
    return 1;
  }
  else
  {
    if ( dword_4D04964 && byte_4F07472[0] == 8 || (unsigned int)*(unsigned __int8 *)qword_4F06410 - 48 > 9 )
    {
      sub_6851D0(0x28u);
      dword_4D03CE0 = 1;
    }
    else
    {
      sub_684B30(0x28u, dword_4F07508);
      while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u )
        sub_7B8B50(0x28u, dword_4F07508, v24, v25, v26, v27);
    }
    return 0;
  }
}
