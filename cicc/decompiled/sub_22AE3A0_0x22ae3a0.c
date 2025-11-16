// Function: sub_22AE3A0
// Address: 0x22ae3a0
//
void __fastcall sub_22AE3A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  _DWORD *v6; // r9
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  char **v12; // r10
  char **v13; // r11
  __int64 v14; // r15
  char **v15; // r13
  int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // [rsp+0h] [rbp-A0h]
  char **v24; // [rsp+8h] [rbp-98h]
  char **v25; // [rsp+10h] [rbp-90h]
  __int64 v26; // [rsp+18h] [rbp-88h]
  int v27; // [rsp+20h] [rbp-80h]
  char *v28[2]; // [rsp+28h] [rbp-78h] BYREF
  _BYTE v29[104]; // [rsp+38h] [rbp-68h] BYREF

  while ( 1 )
  {
    v26 = a3;
    if ( !a4 )
      break;
    v5 = a5;
    if ( !a5 )
      break;
    v6 = (_DWORD *)a1;
    v7 = a4;
    if ( a4 + a5 == 2 )
    {
      v16 = *(_DWORD *)a2;
      if ( *(_DWORD *)a2 > *(_DWORD *)a1 )
      {
        v27 = *(_DWORD *)a1;
        v17 = *(unsigned int *)(a1 + 16);
        v18 = 0xC00000000LL;
        v28[0] = v29;
        v28[1] = (char *)0xC00000000LL;
        if ( (_DWORD)v17 )
        {
          sub_22AD4A0((__int64)v28, (char **)(a1 + 8), v17, 0xC00000000LL, a5, a1);
          v16 = *(_DWORD *)a2;
          v6 = (_DWORD *)a1;
        }
        *v6 = v16;
        sub_22AD4A0(a1 + 8, (char **)(a2 + 8), v17, v18, a5, (__int64)v6);
        *(_DWORD *)a2 = v27;
        sub_22AD4A0(a2 + 8, v28, v19, v20, v21, v22);
        if ( v28[0] != v29 )
          _libc_free((unsigned __int64)v28[0]);
      }
      return;
    }
    if ( a4 > a5 )
    {
      v14 = a4 / 2;
      v12 = (char **)sub_22AD870((_DWORD *)a2, a3, (unsigned int *)(a1 + 72 * (a4 / 2)));
      v8 = 0x8E38E38E38E38E39LL * (((__int64)v12 - a2) >> 3);
    }
    else
    {
      v8 = a5 / 2;
      v13 = (char **)sub_22AD8D0((_DWORD *)a1, a2, (unsigned int *)(a2 + 72 * (a5 / 2)));
      v14 = 0x8E38E38E38E38E39LL * (((__int64)v13 - a1) >> 3);
    }
    v24 = v12;
    v23 = v11;
    v25 = v13;
    v15 = sub_22AE020(v13, (char **)a2, v12, v9, v10, v11);
    sub_22AE3A0(v23, v25, v15, v14, v8);
    a4 = v7 - v14;
    a1 = (__int64)v15;
    a3 = v26;
    a5 = v5 - v8;
    a2 = (__int64)v24;
  }
}
