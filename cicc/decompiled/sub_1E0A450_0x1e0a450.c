// Function: sub_1E0A450
// Address: 0x1e0a450
//
__int64 __fastcall sub_1E0A450(__int64 a1, unsigned int a2, __int64 a3, char a4)
{
  __int64 v7; // rax
  char *v8; // r9
  __int64 v9; // r8
  size_t v10; // rax
  size_t v11; // rsi
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rdi
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // r12
  unsigned int v18; // eax
  __int64 v19; // rdx
  size_t v20; // rsi
  char *v21; // [rsp+8h] [rbp-C8h]
  size_t v22; // [rsp+10h] [rbp-C0h]
  _QWORD v24[2]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v25; // [rsp+30h] [rbp-A0h]
  size_t v26; // [rsp+38h] [rbp-98h]
  int v27; // [rsp+40h] [rbp-90h]
  unsigned __int64 *v28; // [rsp+48h] [rbp-88h]
  unsigned __int64 v29[2]; // [rsp+50h] [rbp-80h] BYREF
  _BYTE v30[112]; // [rsp+60h] [rbp-70h] BYREF

  v7 = sub_1E0A0C0(a1);
  if ( a4 )
  {
    v8 = "l";
    v9 = *(_DWORD *)(v7 + 16) == 2;
    if ( *(_DWORD *)(v7 + 16) != 2 )
      v8 = (char *)byte_3F871B3;
  }
  else
  {
    switch ( *(_DWORD *)(v7 + 16) )
    {
      case 0:
        v9 = 0;
        v8 = (char *)byte_3F871B3;
        break;
      case 1:
      case 3:
        v9 = 2;
        v8 = ".L";
        break;
      case 2:
      case 4:
        v9 = 1;
        v8 = "L";
        break;
      case 5:
        v9 = 1;
        v8 = "$";
        break;
    }
  }
  v21 = v8;
  v22 = v9;
  v29[1] = 0x3C00000000LL;
  v29[0] = (unsigned __int64)v30;
  v27 = 1;
  v24[0] = &unk_49EFC48;
  v26 = 0;
  v25 = 0;
  v24[1] = 0;
  v28 = v29;
  sub_16E7A40((__int64)v24, 0, 0, 0);
  v10 = v26;
  v11 = v26;
  v12 = v25 - v26;
  if ( v25 - v26 < v22 )
  {
    v13 = (_QWORD *)sub_16E7EE0((__int64)v24, v21, v22);
    v10 = v13[3];
    v12 = v13[2] - v10;
  }
  else
  {
    v13 = v24;
    if ( v22 )
    {
      v18 = 0;
      do
      {
        v19 = v18++;
        *(_BYTE *)(v11 + v19) = v21[v19];
      }
      while ( v18 < (unsigned int)v22 );
      v20 = v26;
      v10 = v26 + v22;
      v13 = v24;
      v26 += v22;
      if ( v25 - (v20 + v22) > 2 )
        goto LABEL_7;
      goto LABEL_16;
    }
  }
  if ( v12 > 2 )
  {
LABEL_7:
    *(_BYTE *)(v10 + 2) = 73;
    *(_WORD *)v10 = 21578;
    v13[3] += 3LL;
    goto LABEL_8;
  }
LABEL_16:
  v13 = (_QWORD *)sub_16E7EE0((__int64)v13, "JTI", 3u);
LABEL_8:
  v14 = sub_16E7A90((__int64)v13, *(unsigned int *)(a1 + 336));
  v15 = *(_BYTE **)(v14 + 24);
  if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
  {
    v14 = sub_16E7DE0(v14, 95);
  }
  else
  {
    *(_QWORD *)(v14 + 24) = v15 + 1;
    *v15 = 95;
  }
  sub_16E7A90(v14, a2);
  v24[0] = &unk_49EFD28;
  sub_16E7960((__int64)v24);
  LOWORD(v25) = 262;
  v24[0] = v29;
  v16 = sub_38BF510(a3, v24);
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free(v29[0]);
  return v16;
}
