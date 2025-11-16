// Function: sub_1B28490
// Address: 0x1b28490
//
__int64 __fastcall sub_1B28490(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r13
  __int64 v3; // r15
  const char *v4; // rax
  unsigned __int64 v5; // rdx
  _QWORD *j; // r13
  __int64 v7; // r15
  const char *v8; // rax
  unsigned __int64 v9; // rdx
  _QWORD *k; // r13
  __int64 v11; // r15
  const char *v12; // rax
  unsigned __int64 v13; // rdx
  _QWORD *m; // r15
  __int64 v15; // rdi
  const char *v16; // rax
  unsigned __int64 v17; // rdx
  int *v19; // rax
  size_t v20; // rdx
  int *v21; // rax
  size_t v22; // rdx
  int *v23; // rax
  size_t v24; // rdx
  int *v25; // rax
  size_t v26; // rdx
  char i; // [rsp+Fh] [rbp-141h]
  _DWORD v29[4]; // [rsp+10h] [rbp-140h] BYREF
  _QWORD v30[2]; // [rsp+20h] [rbp-130h] BYREF
  _QWORD v31[2]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v32; // [rsp+40h] [rbp-110h]
  _BYTE *v33; // [rsp+50h] [rbp-100h] BYREF
  __int64 v34; // [rsp+58h] [rbp-F8h]
  _BYTE v35[32]; // [rsp+60h] [rbp-F0h] BYREF
  int v36[52]; // [rsp+80h] [rbp-D0h] BYREF

  sub_16C1840(v36);
  v2 = (_QWORD *)a2[4];
  for ( i = 0; a2 + 3 != v2; v2 = (_QWORD *)v2[1] )
  {
    v3 = 0;
    if ( v2 )
      v3 = (__int64)(v2 - 7);
    if ( !sub_15E4F60(v3) )
    {
      v4 = sub_1649960(v3);
      if ( (v5 <= 4 || *(_DWORD *)v4 != 1836477548 || v4[4] != 46)
        && (*(_BYTE *)(v3 + 32) & 0xF) == 0
        && !sub_15E4F10(v3) )
      {
        v25 = (int *)sub_1649960(v3);
        sub_16C1A90(v36, v25, v26);
        LOBYTE(v33) = 0;
        sub_16C1870(v36, (int *)&v33, 1u);
        i = 1;
      }
    }
  }
  for ( j = (_QWORD *)a2[2]; a2 + 1 != j; j = (_QWORD *)j[1] )
  {
    v7 = 0;
    if ( j )
      v7 = (__int64)(j - 7);
    if ( !sub_15E4F60(v7) )
    {
      v8 = sub_1649960(v7);
      if ( (v9 <= 4 || *(_DWORD *)v8 != 1836477548 || v8[4] != 46)
        && (*(_BYTE *)(v7 + 32) & 0xF) == 0
        && !sub_15E4F10(v7) )
      {
        v23 = (int *)sub_1649960(v7);
        sub_16C1A90(v36, v23, v24);
        LOBYTE(v33) = 0;
        sub_16C1870(v36, (int *)&v33, 1u);
        i = 1;
      }
    }
  }
  for ( k = (_QWORD *)a2[6]; a2 + 5 != k; k = (_QWORD *)k[1] )
  {
    v11 = 0;
    if ( k )
      v11 = (__int64)(k - 6);
    if ( !sub_15E4F60(v11) )
    {
      v12 = sub_1649960(v11);
      if ( (v13 <= 4 || *(_DWORD *)v12 != 1836477548 || v12[4] != 46)
        && (*(_BYTE *)(v11 + 32) & 0xF) == 0
        && !sub_15E4F10(v11) )
      {
        v21 = (int *)sub_1649960(v11);
        sub_16C1A90(v36, v21, v22);
        LOBYTE(v33) = 0;
        sub_16C1870(v36, (int *)&v33, 1u);
        i = 1;
      }
    }
  }
  for ( m = (_QWORD *)a2[8]; a2 + 7 != m; m = (_QWORD *)m[1] )
  {
    v15 = (__int64)(m - 6);
    if ( !m )
      v15 = 0;
    if ( !sub_15E4F60(v15) )
    {
      v16 = sub_1649960(v15);
      if ( (v17 <= 4 || *(_DWORD *)v16 != 1836477548 || v16[4] != 46)
        && (*(_BYTE *)(v15 + 32) & 0xF) == 0
        && !sub_15E4F10(v15) )
      {
        v19 = (int *)sub_1649960(v15);
        sub_16C1A90(v36, v19, v20);
        LOBYTE(v33) = 0;
        sub_16C1870(v36, (int *)&v33, 1u);
        i = 1;
      }
    }
  }
  if ( i )
  {
    sub_16C1AA0(v36, v29);
    v34 = 0x2000000000LL;
    v33 = v35;
    sub_16C1D70((char *)v29, (__int64)&v33);
    v30[0] = v33;
    v30[1] = (unsigned int)v34;
    v32 = 1283;
    v31[0] = "$";
    v31[1] = v30;
    sub_16E2FC0((__int64 *)a1, (__int64)v31);
    if ( v33 != v35 )
      _libc_free((unsigned __int64)v33);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
  }
  return a1;
}
