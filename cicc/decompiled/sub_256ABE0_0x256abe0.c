// Function: sub_256ABE0
// Address: 0x256abe0
//
__int64 __fastcall sub_256ABE0(__int64 a1, __int64 a2)
{
  __int64 v4; // r10
  __int64 v5; // r14
  int v6; // r13d
  unsigned int v7; // r9d
  __int64 v8; // rdx
  __int64 v9; // r15
  int v10; // r10d
  bool v11; // r8
  int v12; // eax
  __int64 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 *v16; // rax
  __int64 *v17; // rdi
  char v18; // al
  __int64 *v19; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r13
  unsigned int v26; // esi
  int v27; // eax
  __int64 *v28; // rdx
  int v29; // eax
  unsigned int v30; // [rsp+10h] [rbp-A0h]
  bool v31; // [rsp+10h] [rbp-A0h]
  int v32; // [rsp+14h] [rbp-9Ch]
  unsigned int v33; // [rsp+14h] [rbp-9Ch]
  __int64 v34; // [rsp+18h] [rbp-98h]
  int v35; // [rsp+18h] [rbp-98h]
  __int64 v36; // [rsp+20h] [rbp-90h]
  __int64 v37; // [rsp+38h] [rbp-78h] BYREF
  __int64 *v38; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v39; // [rsp+48h] [rbp-68h]
  __int64 v40; // [rsp+50h] [rbp-60h]
  __int64 v41; // [rsp+58h] [rbp-58h]
  __int64 *v42[10]; // [rsp+60h] [rbp-50h] BYREF

  v4 = *(unsigned int *)(a1 + 232);
  v5 = *(_QWORD *)(a1 + 216);
  if ( (_DWORD)v4 )
  {
    v6 = v4 - 1;
    if ( a2 )
    {
      v7 = v6 & sub_253B7A0(a2);
      v8 = v5 + 8LL * v7;
    }
    else
    {
      v8 = *(_QWORD *)(a1 + 216);
      v7 = 0;
    }
    v9 = *(_QWORD *)v8;
    if ( a2 != *(_QWORD *)v8 )
    {
      v10 = 1;
      v11 = a2 == -8192 || a2 == -4096;
      while ( 1 )
      {
        if ( v9 != -4096 && v9 != -8192 && !v11 )
        {
          if ( a2 )
          {
            v12 = *(_DWORD *)(a2 + 20) - *(_DWORD *)(a2 + 24);
            if ( v9 )
            {
              if ( v12 == *(_DWORD *)(v9 + 20) - *(_DWORD *)(v9 + 24) )
              {
                if ( !v12 )
                  break;
                v30 = v7;
                v32 = v10;
                v34 = v8;
                v13 = *(__int64 **)(a2 + 8);
                v39 = (__int64 *)sub_254BB00(a2);
                v38 = v13;
                sub_254BBF0((__int64)&v38);
                v14 = *(_QWORD *)a2;
                v40 = a2;
                v41 = v14;
                v42[0] = (__int64 *)sub_254BB00(a2);
                v42[1] = v42[0];
                sub_254BBF0((__int64)v42);
                v16 = *(__int64 **)a2;
                v17 = v38;
                v42[2] = (__int64 *)a2;
                v8 = v34;
                v42[3] = v16;
                v10 = v32;
                v7 = v30;
                v11 = 0;
                if ( v38 == v42[0] )
                  break;
                while ( 1 )
                {
                  v31 = v11;
                  v33 = v7;
                  v35 = v10;
                  v36 = v8;
                  v18 = sub_B19060(v9, *v17, v8, v15);
                  v8 = v36;
                  v10 = v35;
                  v7 = v33;
                  v11 = v31;
                  if ( !v18 )
                    break;
                  v17 = v39;
                  v19 = v38 + 1;
                  v38 = v19;
                  if ( v19 == v39 )
                  {
LABEL_17:
                    if ( v42[0] == v39 )
                      goto LABEL_18;
                  }
                  else
                  {
                    while ( 1 )
                    {
                      v15 = *v19;
                      if ( (unsigned __int64)(*v19 + 2) > 1 )
                        break;
                      v38 = ++v19;
                      if ( v19 == v39 )
                        goto LABEL_17;
                    }
                    v17 = v38;
                    if ( v42[0] == v38 )
                      goto LABEL_18;
                  }
                }
              }
            }
            else if ( !v12 )
            {
              break;
            }
          }
          else if ( !v9 || *(_DWORD *)(v9 + 24) == *(_DWORD *)(v9 + 20) )
          {
            break;
          }
        }
        if ( *(_QWORD *)v8 == -4096 )
          goto LABEL_26;
        v7 = v6 & (v10 + v7);
        v8 = v5 + 8LL * v7;
        v9 = *(_QWORD *)v8;
        if ( a2 == *(_QWORD *)v8 )
          break;
        ++v10;
      }
LABEL_18:
      v5 = *(_QWORD *)(a1 + 216);
      v4 = *(unsigned int *)(a1 + 232);
    }
    if ( v8 != v5 + 8 * v4 )
      return *(_QWORD *)v8;
  }
LABEL_26:
  v21 = sub_A777F0(0x40u, *(__int64 **)(a1 + 112));
  v25 = v21;
  if ( v21 )
    sub_C8CD80(v21, v21 + 32, a2, v22, v23, v24);
  v37 = v25;
  if ( !(unsigned __int8)sub_255F790(a1 + 208, &v37, &v38) )
  {
    v26 = *(_DWORD *)(a1 + 232);
    v27 = *(_DWORD *)(a1 + 224);
    v28 = v38;
    ++*(_QWORD *)(a1 + 208);
    v29 = v27 + 1;
    v42[0] = v28;
    if ( 4 * v29 >= 3 * v26 )
    {
      v26 *= 2;
    }
    else if ( v26 - *(_DWORD *)(a1 + 228) - v29 > v26 >> 3 )
    {
LABEL_37:
      *(_DWORD *)(a1 + 224) = v29;
      if ( *v28 != -4096 )
        --*(_DWORD *)(a1 + 228);
      *v28 = v37;
      return v37;
    }
    sub_256AAA0(a1 + 208, v26);
    sub_255F790(a1 + 208, &v37, v42);
    v28 = v42[0];
    v29 = *(_DWORD *)(a1 + 224) + 1;
    goto LABEL_37;
  }
  return v37;
}
