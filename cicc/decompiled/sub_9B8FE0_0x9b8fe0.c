// Function: sub_9B8FE0
// Address: 0x9b8fe0
//
__int64 __fastcall sub_9B8FE0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // [rsp+8h] [rbp-F8h]
  unsigned int *v16; // [rsp+18h] [rbp-E8h]
  __int64 v17; // [rsp+28h] [rbp-D8h]
  char v19[8]; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v20; // [rsp+48h] [rbp-B8h]
  unsigned int v21; // [rsp+58h] [rbp-A8h]
  char v22[8]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v23; // [rsp+68h] [rbp-98h]
  unsigned int v24; // [rsp+78h] [rbp-88h]
  _QWORD v25[2]; // [rsp+80h] [rbp-80h] BYREF
  _BYTE v26[112]; // [rsp+90h] [rbp-70h] BYREF

  v15 = a3;
  if ( a3 )
  {
    v5 = *a2;
    v6 = 1;
    v25[0] = v26;
    v25[1] = 0x400000000LL;
    sub_B9A9D0(v5, v25);
    v17 = a1;
    v16 = (unsigned int *)&unk_3F1FE44;
    v7 = v5;
LABEL_8:
    if ( (*(_BYTE *)(v7 + 7) & 0x20) != 0 )
    {
      v8 = sub_B91C10(v7, v6);
      if ( v8 && v15 != 1 )
      {
        v9 = 2;
        v10 = v8;
        v11 = a2[1];
        while ( 1 )
        {
          v12 = 0;
          if ( (*(_BYTE *)(v11 + 7) & 0x20) != 0 )
            v12 = sub_B91C10(v11, v6);
          switch ( v6 )
          {
            case 1u:
              v10 = sub_E01DF0(v10, v12);
              goto LABEL_16;
            case 3u:
              v10 = sub_B916B0(v10, v12);
              if ( !v10 )
                goto LABEL_20;
              goto LABEL_17;
            case 6u:
            case 8u:
            case 9u:
              v10 = sub_BA74A0(v10, v12);
              goto LABEL_16;
            case 7u:
              v10 = sub_BA6CD0(v10, v12);
              goto LABEL_16;
            case 0x19u:
              v10 = sub_9B8B30(v17, v11);
              goto LABEL_16;
            case 0x28u:
              sub_B8DF90(v22, v12);
              sub_B8DF90(v19, v10);
              v14 = sub_BD5C60(v17, v10, v13);
              v10 = sub_B8D9F0(v14, v19, v22);
              sub_C7D6A0(v20, 32LL * v21, 8);
              sub_C7D6A0(v23, 32LL * v24, 8);
LABEL_16:
              if ( !v10 )
                goto LABEL_20;
LABEL_17:
              if ( v15 == (_DWORD)v9 )
              {
LABEL_20:
                v8 = v10;
                goto LABEL_6;
              }
              v11 = a2[v9++];
              break;
            default:
              goto LABEL_28;
          }
        }
      }
    }
    else
    {
      v8 = 0;
    }
LABEL_6:
    while ( 1 )
    {
      sub_B99FD0(v17, v6, v8);
      if ( v16 == (unsigned int *)&unk_3F1FE60 )
        break;
      v6 = *v16++;
      v7 = v5;
      if ( v6 )
        goto LABEL_8;
      v8 = *(_QWORD *)(v5 + 48);
      if ( v8 && v15 != 1 )
LABEL_28:
        BUG();
    }
    a1 = v17;
    if ( (_BYTE *)v25[0] != v26 )
      _libc_free(v25[0], v6);
  }
  return a1;
}
