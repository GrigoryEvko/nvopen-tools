// Function: sub_161AE40
// Address: 0x161ae40
//
__int64 __fastcall sub_161AE40(_QWORD *a1, void *a2, void *a3, size_t a4, void *a5, size_t a6)
{
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // r15
  void *v9; // rcx
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 result; // rax
  int v14; // eax
  _BYTE *v15; // rsi
  int v16; // eax
  _BYTE *v17; // rsi
  void *v18; // [rsp+0h] [rbp-A0h]
  _DWORD v19[4]; // [rsp+Ch] [rbp-94h] BYREF
  char v20; // [rsp+1Fh] [rbp-81h] BYREF
  void *s2; // [rsp+20h] [rbp-80h] BYREF
  size_t n; // [rsp+28h] [rbp-78h]
  _QWORD v23[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v24; // [rsp+40h] [rbp-60h]
  _QWORD v25[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v26; // [rsp+60h] [rbp-40h]

  v6 = a1;
  v7 = a1[27];
  v19[0] = (_DWORD)a2;
  v20 = 0;
  if ( *(_QWORD *)(v7 + 32) )
  {
    s2 = a5;
    n = a6;
  }
  else
  {
    s2 = a3;
    n = a4;
  }
  v8 = *((unsigned int *)a1 + 58);
  if ( *((_DWORD *)a1 + 58) )
  {
    v9 = s2;
    v10 = 0;
    v11 = a1[28];
    while ( 1 )
    {
      if ( *(_QWORD *)(v11 + 8) == n )
      {
        if ( !n )
          break;
        a1 = *(_QWORD **)v11;
        a2 = v9;
        v18 = v9;
        v14 = memcmp(*(const void **)v11, v9, n);
        v9 = v18;
        if ( !v14 )
          break;
      }
      ++v10;
      v11 += 48;
      if ( v8 == v10 )
        goto LABEL_7;
    }
    v20 = *(_BYTE *)(v11 + 40);
  }
  else
  {
LABEL_7:
    v12 = sub_16E8CB0(a1, a2, a3);
    v26 = 770;
    v24 = 1283;
    v23[0] = "Cannot find option named '";
    v23[1] = &s2;
    v25[0] = v23;
    v25[1] = "'!";
    result = sub_16B1F90(v6, v25, 0, 0, v12);
    if ( (_BYTE)result )
      return result;
  }
  v15 = (_BYTE *)v6[21];
  if ( v15 == (_BYTE *)v6[22] )
  {
    sub_931B30((__int64)(v6 + 20), v15, &v20);
  }
  else
  {
    if ( v15 )
    {
      *v15 = v20;
      v15 = (_BYTE *)v6[21];
    }
    v6[21] = v15 + 1;
  }
  v16 = v19[0];
  v17 = (_BYTE *)v6[24];
  *((_DWORD *)v6 + 4) = v19[0];
  if ( v17 == (_BYTE *)v6[25] )
  {
    sub_B8BBF0((__int64)(v6 + 23), v17, v19);
    return 0;
  }
  else
  {
    if ( v17 )
    {
      *(_DWORD *)v17 = v16;
      v17 = (_BYTE *)v6[24];
    }
    v6[24] = v17 + 4;
    return 0;
  }
}
