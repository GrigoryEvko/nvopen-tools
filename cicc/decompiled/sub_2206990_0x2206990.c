// Function: sub_2206990
// Address: 0x2206990
//
__int64 __fastcall sub_2206990(__int64 a1, __int64 *a2)
{
  _QWORD *v3; // rdi
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rdx
  unsigned int v8; // r15d
  unsigned int v9; // r14d
  int v10; // r8d
  int v11; // r9d
  char v12; // dl
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 *v15; // rax
  __int64 *v16; // rcx
  __int64 v17; // rax
  int v19; // ecx
  int v20; // ecx
  __int64 v21; // r9
  unsigned int v22; // edx
  int v23; // r8d
  int i; // r10d
  _QWORD *v25; // [rsp+10h] [rbp-170h] BYREF
  __int64 v26; // [rsp+18h] [rbp-168h]
  _QWORD v27[16]; // [rsp+20h] [rbp-160h] BYREF
  __int64 v28; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 *v29; // [rsp+A8h] [rbp-D8h]
  __int64 *v30; // [rsp+B0h] [rbp-D0h]
  __int64 v31; // [rsp+B8h] [rbp-C8h]
  int v32; // [rsp+C0h] [rbp-C0h]
  _QWORD v33[23]; // [rsp+C8h] [rbp-B8h] BYREF

  v3 = v27;
  v29 = v33;
  v30 = v33;
  v25 = v27;
  v27[0] = a2;
  v32 = 0;
  v33[0] = a2;
  v28 = 1;
  v26 = 0x1000000001LL;
  v31 = 0x100000010LL;
  v4 = 1;
  while ( 1 )
  {
    v5 = v4--;
    v6 = v3[v5 - 1];
    LODWORD(v26) = v4;
    v7 = *(_QWORD *)(v6 + 32);
    if ( *(_BYTE *)v7 )
      break;
    LOBYTE(a2) = **(_WORD **)(v6 + 16) == 45 || **(_WORD **)(v6 + 16) == 0;
    v8 = (unsigned int)a2;
    if ( (_BYTE)a2 )
    {
      v9 = *(_DWORD *)(v6 + 40) - 1;
      if ( v9 <= 1 )
        goto LABEL_22;
      while ( 2 )
      {
        v13 = v7 + 40LL * (v9 - 1);
        if ( *(_BYTE *)v13 )
        {
          v3 = v25;
          v8 = 0;
          goto LABEL_24;
        }
        v14 = sub_1E69D00(*(_QWORD *)(a1 + 232), *(_DWORD *)(v13 + 8));
        v15 = v29;
        if ( v30 != v29 )
          goto LABEL_6;
        a2 = &v29[HIDWORD(v31)];
        if ( v29 != a2 )
        {
          v16 = 0;
          while ( v14 != *v15 )
          {
            if ( *v15 == -2 )
              v16 = v15;
            if ( a2 == ++v15 )
            {
              if ( !v16 )
                goto LABEL_29;
              *v16 = v14;
              --v32;
              ++v28;
              goto LABEL_19;
            }
          }
          goto LABEL_7;
        }
LABEL_29:
        if ( HIDWORD(v31) < (unsigned int)v31 )
        {
          ++HIDWORD(v31);
          *a2 = v14;
          v17 = (unsigned int)v26;
          ++v28;
          if ( (unsigned int)v26 >= HIDWORD(v26) )
          {
LABEL_31:
            a2 = v27;
            sub_16CD150((__int64)&v25, v27, 0, 8, v10, v11);
            v17 = (unsigned int)v26;
          }
LABEL_20:
          v9 -= 2;
          v25[v17] = v14;
          LODWORD(v26) = v26 + 1;
          if ( v9 <= 1 )
          {
LABEL_21:
            v4 = v26;
            v3 = v25;
            goto LABEL_22;
          }
        }
        else
        {
LABEL_6:
          LODWORD(a2) = v14;
          sub_16CCBA0((__int64)&v28, v14);
          if ( v12 )
          {
LABEL_19:
            v17 = (unsigned int)v26;
            if ( (unsigned int)v26 >= HIDWORD(v26) )
              goto LABEL_31;
            goto LABEL_20;
          }
LABEL_7:
          v9 -= 2;
          if ( v9 <= 1 )
            goto LABEL_21;
        }
        v7 = *(_QWORD *)(v6 + 32);
        continue;
      }
    }
    v19 = *(_DWORD *)(a1 + 336);
    if ( !v19 )
      goto LABEL_24;
    LODWORD(a2) = *(_DWORD *)(v7 + 8);
    v20 = v19 - 1;
    v21 = *(_QWORD *)(a1 + 320);
    v22 = v20 & (37 * (_DWORD)a2);
    v23 = *(_DWORD *)(v21 + 4LL * v22);
    if ( (_DWORD)a2 != v23 )
    {
      for ( i = 1; ; ++i )
      {
        if ( v23 == -1 )
          goto LABEL_24;
        v22 = v20 & (i + v22);
        v23 = *(_DWORD *)(v21 + 4LL * v22);
        if ( (_DWORD)a2 == v23 )
          break;
      }
    }
LABEL_22:
    if ( !v4 )
    {
      v8 = 1;
      goto LABEL_24;
    }
  }
  v8 = 0;
LABEL_24:
  if ( v3 != v27 )
    _libc_free((unsigned __int64)v3);
  if ( v30 != v29 )
    _libc_free((unsigned __int64)v30);
  return v8;
}
