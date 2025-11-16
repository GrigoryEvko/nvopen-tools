// Function: sub_8988D0
// Address: 0x8988d0
//
char *__fastcall sub_8988D0(__int64 a1, unsigned int *a2)
{
  __int64 v2; // r14
  _QWORD *v3; // r13
  char *v4; // r15
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v16; // r8
  __int64 v17; // r9
  int v18; // edx
  int v19; // [rsp+8h] [rbp-38h] BYREF
  int v20[13]; // [rsp+Ch] [rbp-34h] BYREF

  v2 = *(_QWORD *)(a1 + 88);
  v19 = 0;
  v3 = **(_QWORD ***)(v2 + 32);
  if ( !(_DWORD)a2 )
    sub_7296C0(v20);
  v4 = (char *)sub_726700(32);
  v5 = sub_72C390();
  v4[27] |= 4u;
  *(_QWORD *)v4 = v5;
  v6 = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)(v4 + 28) = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)(v4 + 36) = v6;
  *(_QWORD *)(v4 + 44) = qword_4F063F0;
  *((_QWORD *)v4 + 7) = *(_QWORD *)(v2 + 104);
  sub_7B8B50(0x20u, a2, v7, v8, v9, v10);
  if ( word_4F06418[0] == 43 )
  {
    sub_7B8B50(0x20u, a2, v11, v12, v13, v14);
    ++*(_BYTE *)(qword_4F061C8 + 52LL);
    *((_QWORD *)v4 + 8) = sub_7C7EE0(a1, 1, &v19);
    sub_7BE280(0x2Cu, 439, 0, 0, v16, v17);
    v18 = v19;
    *(_QWORD *)(v4 + 44) = qword_4F063F0;
    --*(_BYTE *)(qword_4F061C8 + 52LL);
    if ( v18 )
      goto LABEL_8;
    if ( !v3 )
      goto LABEL_14;
  }
  else
  {
    if ( !v3 || *v3 && (*(_BYTE *)(*v3 + 56LL) & 0x11) == 0 )
    {
      sub_6851C0(0x1B6u, &dword_4F063F8);
      v19 = 1;
LABEL_8:
      v4 = 0;
      goto LABEL_9;
    }
    if ( v19 )
      goto LABEL_8;
  }
  if ( *(_BYTE *)(v3[1] + 80LL) != 3 )
  {
LABEL_14:
    sub_6854C0(0xC34u, (FILE *)(v4 + 28), a1);
    v19 = 1;
    goto LABEL_8;
  }
LABEL_9:
  if ( !(_DWORD)a2 )
    sub_729730(v20[0]);
  return v4;
}
