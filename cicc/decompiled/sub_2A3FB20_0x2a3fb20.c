// Function: sub_2A3FB20
// Address: 0x2a3fb20
//
__int64 *__fastcall sub_2A3FB20(__int64 *a1, _QWORD *a2)
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
  char i; // [rsp+Fh] [rbp-151h]
  _DWORD v29[4]; // [rsp+10h] [rbp-150h] BYREF
  void *v30[4]; // [rsp+20h] [rbp-140h] BYREF
  __int16 v31; // [rsp+40h] [rbp-120h]
  _BYTE *v32; // [rsp+50h] [rbp-110h] BYREF
  void *v33; // [rsp+58h] [rbp-108h]
  __int64 v34; // [rsp+60h] [rbp-100h]
  _BYTE v35[40]; // [rsp+68h] [rbp-F8h] BYREF
  int v36[52]; // [rsp+90h] [rbp-D0h] BYREF

  sub_C7D030(v36);
  v2 = (_QWORD *)a2[4];
  for ( i = 0; a2 + 3 != v2; v2 = (_QWORD *)v2[1] )
  {
    v3 = 0;
    if ( v2 )
      v3 = (__int64)(v2 - 7);
    if ( !sub_B2FC80(v3) )
    {
      v4 = sub_BD5D20(v3);
      if ( (v5 <= 4 || *(_DWORD *)v4 != 1836477548 || v4[4] != 46)
        && (*(_BYTE *)(v3 + 32) & 0xF) == 0
        && !sub_B326A0(v3) )
      {
        v21 = (int *)sub_BD5D20(v3);
        sub_C7D280(v36, v21, v22);
        LOBYTE(v32) = 0;
        sub_C7D060(v36, (int *)&v32, 1u);
        i = 1;
      }
    }
  }
  for ( j = (_QWORD *)a2[2]; a2 + 1 != j; j = (_QWORD *)j[1] )
  {
    v7 = 0;
    if ( j )
      v7 = (__int64)(j - 7);
    if ( !sub_B2FC80(v7) )
    {
      v8 = sub_BD5D20(v7);
      if ( (v9 <= 4 || *(_DWORD *)v8 != 1836477548 || v8[4] != 46)
        && (*(_BYTE *)(v7 + 32) & 0xF) == 0
        && !sub_B326A0(v7) )
      {
        v19 = (int *)sub_BD5D20(v7);
        sub_C7D280(v36, v19, v20);
        LOBYTE(v32) = 0;
        sub_C7D060(v36, (int *)&v32, 1u);
        i = 1;
      }
    }
  }
  for ( k = (_QWORD *)a2[6]; a2 + 5 != k; k = (_QWORD *)k[1] )
  {
    v11 = 0;
    if ( k )
      v11 = (__int64)(k - 6);
    if ( !sub_B2FC80(v11) )
    {
      v12 = sub_BD5D20(v11);
      if ( (v13 <= 4 || *(_DWORD *)v12 != 1836477548 || v12[4] != 46)
        && (*(_BYTE *)(v11 + 32) & 0xF) == 0
        && !sub_B326A0(v11) )
      {
        v25 = (int *)sub_BD5D20(v11);
        sub_C7D280(v36, v25, v26);
        LOBYTE(v32) = 0;
        sub_C7D060(v36, (int *)&v32, 1u);
        i = 1;
      }
    }
  }
  for ( m = (_QWORD *)a2[8]; a2 + 7 != m; m = (_QWORD *)m[1] )
  {
    v15 = (__int64)(m - 7);
    if ( !m )
      v15 = 0;
    if ( !sub_B2FC80(v15) )
    {
      v16 = sub_BD5D20(v15);
      if ( (v17 <= 4 || *(_DWORD *)v16 != 1836477548 || v16[4] != 46)
        && (*(_BYTE *)(v15 + 32) & 0xF) == 0
        && !sub_B326A0(v15) )
      {
        v23 = (int *)sub_BD5D20(v15);
        sub_C7D280(v36, v23, v24);
        LOBYTE(v32) = 0;
        sub_C7D060(v36, (int *)&v32, 1u);
        i = 1;
      }
    }
  }
  if ( i )
  {
    sub_C7D290(v36, v29);
    v32 = v35;
    v33 = 0;
    v34 = 32;
    sub_C7D4E0((unsigned __int8 *)v29, &v32);
    v31 = 1283;
    v30[0] = ".";
    v30[2] = v32;
    v30[3] = v33;
    sub_CA0F50(a1, v30);
    if ( v32 != v35 )
      _libc_free((unsigned __int64)v32);
  }
  else
  {
    *a1 = (__int64)(a1 + 2);
    sub_2A3E750(a1, byte_3F871B3, (__int64)byte_3F871B3);
  }
  return a1;
}
