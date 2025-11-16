// Function: sub_2550530
// Address: 0x2550530
//
__int64 __fastcall sub_2550530(__int64 a1)
{
  __int64 v2; // r15
  __int64 v3; // rsi
  unsigned __int64 v4; // rdi
  __int64 **v6; // rbx
  __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rsi
  int v11; // edi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 *v15; // rax
  __int64 v16; // [rsp+8h] [rbp-88h]
  __int64 v17; // [rsp+18h] [rbp-78h]
  __int64 v18; // [rsp+18h] [rbp-78h]
  _QWORD v19[4]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v20[10]; // [rsp+40h] [rbp-50h] BYREF

  *(_QWORD *)a1 = &unk_4A1D4C0;
  *(_QWORD *)(a1 + 88) = &unk_4A1D550;
  if ( !*(_DWORD *)(a1 + 192) )
  {
    v2 = *(_QWORD *)(a1 + 176);
    v3 = 0;
    goto LABEL_3;
  }
  if ( !byte_4FEF240[0] && (unsigned int)sub_2207590((__int64)byte_4FEF240) )
  {
    unk_4FEF260 = -4096;
    unk_4FEF268 = -4096;
    unk_4FEF270 = 0;
    unk_4FEF278 = 0;
    sub_2207640((__int64)byte_4FEF240);
  }
  if ( !byte_4FEF208[0] && (unsigned int)sub_2207590((__int64)byte_4FEF208) )
  {
    unk_4FEF220 = -8192;
    unk_4FEF228 = -8192;
    qword_4FEF230 = 0;
    unk_4FEF238 = 0;
    sub_2207640((__int64)byte_4FEF208);
  }
  v6 = *(__int64 ***)(a1 + 176);
  v3 = *(unsigned int *)(a1 + 192);
  v2 = (__int64)&v6[v3];
  if ( v6 != &v6[v3] )
  {
    while ( 1 )
    {
      v7 = *v6;
      v8 = **v6;
      v9 = (*v6)[1];
      if ( unk_4FEF268 != v9 || unk_4FEF260 != v8 )
        goto LABEL_11;
      v10 = v7[2];
      if ( v10 != unk_4FEF270 )
      {
        if ( v10 == -4096 || unk_4FEF270 == -8192 || unk_4FEF270 == -4096 || v10 == -8192 )
          goto LABEL_11;
        if ( v10 )
        {
          v11 = *(_DWORD *)(v10 + 20) - *(_DWORD *)(v10 + 24);
          if ( unk_4FEF270 )
          {
            v17 = unk_4FEF270;
            if ( v11 != *(_DWORD *)(unk_4FEF270 + 20LL) - *(_DWORD *)(unk_4FEF270 + 24LL) )
              goto LABEL_11;
            if ( v11 )
            {
              v16 = *(_QWORD *)(v10 + 8);
              v19[1] = sub_254BB00(v10);
              v19[0] = v16;
              sub_254BBF0((__int64)v19);
              v19[2] = v10;
              v19[3] = *(_QWORD *)v10;
              v20[0] = sub_254BB00(v10);
              v20[1] = v20[0];
              sub_254BBF0((__int64)v20);
              v14 = v17;
              v20[2] = v10;
              v20[3] = *(_QWORD *)v10;
              v15 = (__int64 *)v19[0];
              if ( v20[0] != v19[0] )
              {
                while ( 1 )
                {
                  v18 = v14;
                  if ( !(unsigned __int8)sub_B19060(v14, *v15, v12, v13) )
                    break;
                  v19[0] += 8LL;
                  sub_254BBF0((__int64)v19);
                  v15 = (__int64 *)v19[0];
                  v14 = v18;
                  if ( v19[0] == v20[0] )
                    goto LABEL_14;
                }
                v7 = *v6;
                v8 = **v6;
                v9 = (*v6)[1];
LABEL_11:
                if ( unk_4FEF228 == v9 && unk_4FEF220 == v8 )
                  sub_254C7C0((__int64 *)v7[2], qword_4FEF230);
              }
            }
          }
          else if ( v11 )
          {
            goto LABEL_11;
          }
        }
        else if ( unk_4FEF270 && *(_DWORD *)(unk_4FEF270 + 24LL) != *(_DWORD *)(unk_4FEF270 + 20LL) )
        {
          goto LABEL_11;
        }
      }
LABEL_14:
      if ( (__int64 **)v2 == ++v6 )
      {
        v2 = *(_QWORD *)(a1 + 176);
        v3 = *(unsigned int *)(a1 + 192);
        break;
      }
    }
  }
LABEL_3:
  sub_C7D6A0(v2, v3 * 8, 8);
  v4 = *(_QWORD *)(a1 + 104);
  if ( v4 != a1 + 120 )
    _libc_free(v4);
  *(_QWORD *)a1 = &unk_4A16C00;
  return sub_254FD20(a1 + 8);
}
