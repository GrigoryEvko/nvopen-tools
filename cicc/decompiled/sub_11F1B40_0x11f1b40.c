// Function: sub_11F1B40
// Address: 0x11f1b40
//
__int64 __fastcall sub_11F1B40(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r15
  unsigned __int64 v10; // r15
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rdi
  unsigned int v17; // r12d
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *); // rax
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int *v22; // rbx
  __int64 v23; // r12
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rsi
  __int64 v30; // [rsp+18h] [rbp-98h] BYREF
  _QWORD v31[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v32; // [rsp+40h] [rbp-70h]
  _QWORD *v33[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v34; // [rsp+70h] [rbp-40h]

  v6 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v33[0] = &v30;
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 || (v10 = *(_QWORD *)(v7 + 8)) != 0 || !(unsigned __int8)sub_995E90(v33, v6, a3, a4, a5) )
  {
    if ( a2 )
    {
      if ( *(_BYTE *)v6 == 85 )
      {
        v26 = *(_QWORD *)(v6 - 32);
        if ( v26 )
        {
          if ( !*(_BYTE *)v26 && *(_QWORD *)(v26 + 24) == *(_QWORD *)(v6 + 80) && *(_DWORD *)(v26 + 36) == 170 )
          {
            v27 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
            if ( v27 )
            {
              v30 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
              goto LABEL_35;
            }
          }
          if ( !*(_BYTE *)v26 && *(_QWORD *)(v26 + 24) == *(_QWORD *)(v6 + 80) && *(_DWORD *)(v26 + 36) == 26 )
          {
            v27 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
            if ( v27 )
            {
              v30 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
              if ( *(_BYTE *)v6 == 85 )
              {
LABEL_35:
                v28 = *(_QWORD *)(a1 - 32);
                v31[0] = v27;
                v34 = 257;
                if ( v28 )
                {
                  if ( !*(_BYTE *)v28 )
                  {
                    v29 = *(_QWORD *)(a1 + 80);
                    if ( *(_QWORD *)(v28 + 24) == v29 )
                    {
LABEL_40:
                      v8 = sub_921880((unsigned int **)a3, v29, v28, (int)v31, 1, (__int64)v33, 0);
                      sub_B45230(v8, a1);
                      if ( v8 && *(_BYTE *)v8 == 85 )
                        *(_WORD *)(v8 + 2) = *(_WORD *)(v8 + 2) & 0xFFFC | *(_WORD *)(a1 + 2) & 3;
                      return v8;
                    }
                  }
                  LODWORD(v28) = 0;
                }
                v29 = 0;
                goto LABEL_40;
              }
            }
          }
        }
      }
    }
    return 0;
  }
  v11 = *(_QWORD *)(a1 - 32);
  v34 = 257;
  v31[0] = v30;
  if ( v11 )
  {
    if ( !*(_BYTE *)v11 && *(_QWORD *)(v11 + 24) == *(_QWORD *)(a1 + 80) )
      v10 = *(_QWORD *)(a1 + 80);
    else
      LODWORD(v11) = 0;
  }
  v8 = sub_921880((unsigned int **)a3, v10, v11, (int)v31, 1, (__int64)v33, 0);
  sub_B45230(v8, a1);
  if ( v8 && *(_BYTE *)v8 == 85 )
    *(_WORD *)(v8 + 2) = *(_WORD *)(v8 + 2) & 0xFFFC | *(_WORD *)(a1 + 2) & 3;
  if ( !a2 )
  {
    v32 = 257;
    v12 = sub_B45210(a1);
    v16 = *(_QWORD *)(a3 + 80);
    v17 = v12;
    v18 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *))(*(_QWORD *)v16 + 48LL);
    if ( v18 == sub_9288C0 )
    {
      if ( *(_BYTE *)v8 > 0x15u )
        goto LABEL_22;
      v19 = sub_AAAFF0(12, (unsigned __int8 *)v8, v13, v14, v15);
    }
    else
    {
      v19 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))v18)(v16, 12, v8, v17);
    }
    if ( v19 )
      return v19;
LABEL_22:
    v34 = 257;
    v20 = sub_B50340(12, v8, (__int64)v33, 0, 0);
    v21 = *(_QWORD *)(a3 + 96);
    v19 = v20;
    if ( v21 )
      sub_B99FD0(v20, 3u, v21);
    sub_B45150(v19, v17);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v19,
      v31,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64));
    v22 = *(unsigned int **)a3;
    v23 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    if ( *(_QWORD *)a3 != v23 )
    {
      do
      {
        v24 = *((_QWORD *)v22 + 1);
        v25 = *v22;
        v22 += 4;
        sub_B99FD0(v19, v25, v24);
      }
      while ( (unsigned int *)v23 != v22 );
    }
    return v19;
  }
  return v8;
}
