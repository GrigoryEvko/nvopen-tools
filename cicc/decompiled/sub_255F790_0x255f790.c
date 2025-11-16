// Function: sub_255F790
// Address: 0x255f790
//
__int64 __fastcall sub_255F790(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r12d
  __int64 v5; // r13
  __int64 v6; // r9
  int v7; // r12d
  unsigned int v8; // r8d
  __int64 *v9; // r14
  __int64 v10; // rdx
  __int64 *v11; // r10
  int v12; // eax
  __int64 v13; // rcx
  __int64 *v14; // rdi
  __int64 v15; // rdx
  char v16; // al
  __int64 *v17; // rax
  __int64 *v19; // rax
  __int64 *v20; // [rsp+8h] [rbp-A8h]
  unsigned int v21; // [rsp+10h] [rbp-A0h]
  __int64 v22; // [rsp+10h] [rbp-A0h]
  __int64 v23; // [rsp+18h] [rbp-98h]
  unsigned int v24; // [rsp+18h] [rbp-98h]
  __int64 *v25; // [rsp+20h] [rbp-90h]
  __int64 v26; // [rsp+20h] [rbp-90h]
  __int64 v27; // [rsp+28h] [rbp-88h]
  __int64 *v28; // [rsp+28h] [rbp-88h]
  int v29; // [rsp+34h] [rbp-7Ch]
  __int64 *v30; // [rsp+40h] [rbp-70h] BYREF
  __int64 v31; // [rsp+48h] [rbp-68h]
  __int64 v32; // [rsp+50h] [rbp-60h]
  __int64 v33; // [rsp+58h] [rbp-58h]
  _QWORD v34[10]; // [rsp+60h] [rbp-50h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v5 = *a2;
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v4 - 1;
    if ( *a2 )
    {
      v8 = v7 & sub_253B7A0(*a2);
      v9 = (__int64 *)(v6 + 8LL * v8);
    }
    else
    {
      v9 = *(__int64 **)(a1 + 8);
      v8 = 0;
    }
    v10 = *v9;
    if ( *v9 == v5 )
    {
LABEL_18:
      *a3 = v9;
      return 1;
    }
    else
    {
      v29 = 1;
      v11 = 0;
      while ( 1 )
      {
        if ( v5 != -8192 && v5 != -4096 && v10 != -4096 && v10 != -8192 )
        {
          if ( v5 )
          {
            v12 = *(_DWORD *)(v5 + 20) - *(_DWORD *)(v5 + 24);
            if ( v10 )
            {
              v27 = v10;
              if ( v12 == *(_DWORD *)(v10 + 20) - *(_DWORD *)(v10 + 24) )
              {
                v21 = v8;
                v23 = v6;
                v25 = v11;
                if ( !v12 )
                  goto LABEL_18;
                v20 = *(__int64 **)(v5 + 8);
                v31 = sub_254BB00(v5);
                v30 = v20;
                sub_254BBF0((__int64)&v30);
                v32 = v5;
                v33 = *(_QWORD *)v5;
                v34[0] = sub_254BB00(v5);
                v34[1] = v34[0];
                sub_254BBF0((__int64)v34);
                v34[2] = v5;
                v14 = v30;
                v34[3] = *(_QWORD *)v5;
                v11 = v25;
                v6 = v23;
                v8 = v21;
                v15 = v27;
                if ( (__int64 *)v34[0] == v30 )
                  goto LABEL_18;
                while ( 1 )
                {
                  v24 = v8;
                  v26 = v6;
                  v28 = v11;
                  v22 = v15;
                  v16 = sub_B19060(v15, *v14, v15, v13);
                  v11 = v28;
                  v6 = v26;
                  v8 = v24;
                  if ( !v16 )
                    break;
                  v14 = (__int64 *)v31;
                  v15 = v22;
                  v17 = ++v30;
                  if ( v30 != (__int64 *)v31 )
                  {
                    while ( 1 )
                    {
                      v13 = *v17;
                      if ( (unsigned __int64)(*v17 + 2) > 1 )
                        break;
                      v30 = ++v17;
                      if ( v17 == (__int64 *)v31 )
                        goto LABEL_17;
                    }
                    v14 = v30;
                  }
LABEL_17:
                  if ( v14 == (__int64 *)v34[0] )
                    goto LABEL_18;
                }
              }
            }
            else if ( !v12 )
            {
              goto LABEL_18;
            }
          }
          else if ( !v10 || *(_DWORD *)(v10 + 24) == *(_DWORD *)(v10 + 20) )
          {
            goto LABEL_18;
          }
        }
        if ( *v9 == -4096 )
          break;
        if ( *v9 != -8192 || v11 )
          v9 = v11;
        v5 = *a2;
        v8 = v7 & (v29 + v8);
        v19 = (__int64 *)(v6 + 8LL * v8);
        v10 = *v19;
        if ( *v19 == *a2 )
        {
          v9 = (__int64 *)(v6 + 8LL * v8);
          goto LABEL_18;
        }
        v11 = v9;
        ++v29;
        v9 = (__int64 *)(v6 + 8LL * v8);
      }
      if ( !v11 )
        v11 = v9;
      *a3 = v11;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
