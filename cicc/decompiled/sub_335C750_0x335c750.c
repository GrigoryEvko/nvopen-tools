// Function: sub_335C750
// Address: 0x335c750
//
__int64 __fastcall sub_335C750(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v5; // r9
  unsigned int v9; // r8d
  unsigned int *v10; // rax
  unsigned int v11; // eax
  _WORD *v12; // r11
  __int64 v13; // r8
  size_t v14; // r10
  __int64 v15; // rax
  __int64 v16; // rcx
  char *v18; // rdi
  unsigned __int64 v19; // rdx
  char *v20; // rax
  __int64 v22; // [rsp-A8h] [rbp-A8h]
  _WORD *v23; // [rsp-A0h] [rbp-A0h]
  __int64 v24; // [rsp-A0h] [rbp-A0h]
  __int64 v25; // [rsp-98h] [rbp-98h]
  __int64 v26; // [rsp-98h] [rbp-98h]
  __int64 v27; // [rsp-98h] [rbp-98h]
  char *v28; // [rsp-88h] [rbp-88h] BYREF
  __int64 v29; // [rsp-80h] [rbp-80h]
  _BYTE v30[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( a1 != a2 )
  {
    v5 = a3;
    if ( a2 )
    {
      v9 = 0;
      v10 = (unsigned int *)(*(_QWORD *)(a1 + 40) + 40LL * (unsigned int)(*(_DWORD *)(a1 + 64) - 1));
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v10 + 48LL) + 16LL * v10[2]) == 262 )
        return v9;
    }
    v11 = *(_DWORD *)(a1 + 68);
    v12 = *(_WORD **)(a1 + 48);
    v9 = 0;
    if ( v12[8 * v11 - 8] == 262 )
      return v9;
    v13 = v11;
    v28 = v30;
    v14 = 16LL * v11;
    v29 = 0x400000000LL;
    if ( v11 > 4uLL )
    {
      v22 = 16LL * v11;
      v23 = v12;
      v25 = v11;
      sub_C8D5F0((__int64)&v28, v30, v11, 0x10u, v11, a3);
      v13 = v25;
      v12 = v23;
      v14 = v22;
      v5 = a3;
      v18 = &v28[16 * (unsigned int)v29];
    }
    else
    {
      if ( !v14 )
      {
LABEL_7:
        LODWORD(v15) = v13 + v14;
        LODWORD(v29) = v13 + v14;
        v16 = (unsigned int)(v13 + v14);
        if ( a4 )
        {
          v15 = (unsigned int)v15;
          v19 = (unsigned int)v15 + 1LL;
          if ( v19 > HIDWORD(v29) )
          {
            v27 = v5;
            sub_C8D5F0((__int64)&v28, v30, v19, 0x10u, v13, v5);
            v15 = (unsigned int)v29;
            v5 = v27;
          }
          v20 = &v28[16 * v15];
          *(_QWORD *)v20 = 262;
          *((_QWORD *)v20 + 1) = 0;
          v16 = (unsigned int)(v29 + 1);
          LODWORD(v29) = v29 + 1;
        }
        sub_335C390(a1, a5, v28, v16, a2, v5);
        if ( v28 != v30 )
          _libc_free((unsigned __int64)v28);
        return 1;
      }
      v18 = v30;
    }
    v24 = v13;
    v26 = v5;
    memcpy(v18, v12, v14);
    LODWORD(v14) = v29;
    v13 = v24;
    v5 = v26;
    goto LABEL_7;
  }
  return 0;
}
