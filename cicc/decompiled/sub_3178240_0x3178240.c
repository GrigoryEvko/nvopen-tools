// Function: sub_3178240
// Address: 0x3178240
//
__int64 __fastcall sub_3178240(__int64 a1, unsigned int *a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v7; // r14
  _QWORD *v8; // rdi
  __int64 v9; // rsi
  int v10; // r13d
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  int v14; // eax
  int v15; // r8d
  __int64 v16; // rdi
  unsigned int i; // edx
  int *v18; // rax
  int v19; // r9d
  __int64 v20; // rcx
  _QWORD *v21; // r10
  _QWORD *v22; // r11
  _QWORD *v23; // rcx
  unsigned int v24; // edx
  unsigned __int64 v25; // [rsp+0h] [rbp-100h] BYREF
  unsigned __int64 v26; // [rsp+8h] [rbp-F8h] BYREF
  int v27; // [rsp+10h] [rbp-F0h]
  char *v28; // [rsp+18h] [rbp-E8h]
  __int64 v29; // [rsp+20h] [rbp-E0h]
  char v30; // [rsp+28h] [rbp-D8h] BYREF
  int v31; // [rsp+70h] [rbp-90h]
  char *v32; // [rsp+78h] [rbp-88h]
  __int64 v33; // [rsp+80h] [rbp-80h]
  char v34; // [rsp+88h] [rbp-78h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = (_QWORD *)*((_QWORD *)a2 + 1);
  v9 = a2[4];
  v10 = v4 - 1;
  v32 = &v34;
  v28 = &v30;
  v33 = 0x400000000LL;
  v27 = -1;
  v29 = 0x400000000LL;
  v31 = -2;
  v11 = sub_3177C10(v8, &v8[2 * v9]);
  v12 = *a2;
  v26 = v11;
  v13 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * v12)) >> 47)
       ^ (0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * v12)));
  v25 = 0x9DDFEA08EB382D69LL * ((v13 >> 47) ^ v13);
  v14 = sub_C41E80((__int64 *)&v25, (__int64 *)&v26);
  v15 = 1;
  v16 = 0;
  for ( i = v10 & v14; ; i = v10 & v24 )
  {
    v18 = (int *)(v7 + 96LL * i);
    v19 = *v18;
    if ( *a2 == *v18 )
    {
      v20 = a2[4];
      if ( v20 == v18[4] )
      {
        v21 = (_QWORD *)*((_QWORD *)a2 + 1);
        v22 = (_QWORD *)*((_QWORD *)v18 + 1);
        v23 = &v21[2 * v20];
        if ( v21 == v23 )
        {
LABEL_16:
          *a3 = v18;
          return 1;
        }
        while ( *v21 == *v22 && v21[1] == v22[1] )
        {
          v21 += 2;
          v22 += 2;
          if ( v23 == v21 )
            goto LABEL_16;
        }
      }
    }
    if ( v19 == -1 )
      break;
    if ( v19 == -2 && !v18[4] && !v16 )
      v16 = v7 + 96LL * i;
LABEL_18:
    v24 = v15 + i;
    ++v15;
  }
  if ( v18[4] )
    goto LABEL_18;
  if ( !v16 )
    v16 = v7 + 96LL * i;
  *a3 = v16;
  return 0;
}
