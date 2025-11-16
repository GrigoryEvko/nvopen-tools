// Function: sub_2E792C0
// Address: 0x2e792c0
//
__int64 __fastcall sub_2E792C0(__int64 a1, unsigned int a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  char *v7; // r9
  size_t v8; // rbx
  _BYTE *v9; // rax
  _BYTE *v10; // rsi
  unsigned __int64 v11; // rdx
  _BYTE *v12; // rax
  __int64 v13; // r12
  unsigned int v15; // eax
  __int64 v16; // rdx
  _BYTE *v17; // rsi
  unsigned __int8 *v18; // [rsp+0h] [rbp-E0h]
  const char *v20; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v21; // [rsp+18h] [rbp-C8h]
  __int64 v22; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v23; // [rsp+28h] [rbp-B8h]
  _BYTE *v24; // [rsp+30h] [rbp-B0h]
  __int64 v25; // [rsp+38h] [rbp-A8h]
  const char **v26; // [rsp+40h] [rbp-A0h]
  const char *v27; // [rsp+50h] [rbp-90h] BYREF
  __int64 v28; // [rsp+58h] [rbp-88h]
  __int64 v29; // [rsp+60h] [rbp-80h]
  _BYTE v30[120]; // [rsp+68h] [rbp-78h] BYREF

  v6 = sub_2E79000((__int64 *)a1);
  if ( a4 )
  {
    v7 = "l";
    v8 = *(_DWORD *)(v6 + 24) == 2;
    if ( *(_DWORD *)(v6 + 24) != 2 )
      v7 = (char *)byte_3F871B3;
  }
  else
  {
    switch ( *(_DWORD *)(v6 + 24) )
    {
      case 0:
        v8 = 0;
        v7 = (char *)byte_3F871B3;
        break;
      case 1:
      case 3:
        v8 = 2;
        v7 = ".L";
        break;
      case 2:
      case 4:
        v8 = 1;
        v7 = "L";
        break;
      case 5:
        v8 = 2;
        v7 = "L#";
        break;
      case 6:
        v8 = 1;
        v7 = "$";
        break;
      case 7:
        v8 = 3;
        v7 = "L..";
        break;
      default:
        BUG();
    }
  }
  v25 = 0x100000000LL;
  v18 = (unsigned __int8 *)v7;
  v20 = (const char *)&unk_49DD288;
  v26 = &v27;
  v27 = v30;
  v28 = 0;
  v29 = 60;
  v21 = 2;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  sub_CB5980((__int64)&v20, 0, 0, 0);
  v9 = v24;
  v10 = v24;
  v11 = v23 - (_QWORD)v24;
  if ( v23 - (unsigned __int64)v24 < v8 )
  {
    sub_CB6200((__int64)&v20, v18, v8);
    v9 = v24;
    v11 = v23 - (_QWORD)v24;
  }
  else if ( v8 )
  {
    v15 = 0;
    do
    {
      v16 = v15++;
      v10[v16] = v18[v16];
    }
    while ( v15 < (unsigned int)v8 );
    v17 = v24;
    v9 = &v24[v8];
    v24 += v8;
    if ( v23 - (unsigned __int64)&v17[v8] > 2 )
      goto LABEL_7;
    goto LABEL_16;
  }
  if ( v11 > 2 )
  {
LABEL_7:
    v9[2] = 73;
    *(_WORD *)v9 = 21578;
    v24 += 3;
    goto LABEL_8;
  }
LABEL_16:
  sub_CB6200((__int64)&v20, "JTI", 3u);
LABEL_8:
  sub_CB59D0((__int64)&v20, *(unsigned int *)(a1 + 336));
  v12 = v24;
  if ( (unsigned __int64)v24 >= v23 )
  {
    sub_CB5D20((__int64)&v20, 95);
  }
  else
  {
    ++v24;
    *v12 = 95;
  }
  sub_CB59D0((__int64)&v20, a2);
  v20 = (const char *)&unk_49DD388;
  sub_CB5840((__int64)&v20);
  LOWORD(v24) = 261;
  v20 = v27;
  v21 = v28;
  v13 = sub_E6C460(a3, &v20);
  if ( v27 != v30 )
    _libc_free((unsigned __int64)v27);
  return v13;
}
