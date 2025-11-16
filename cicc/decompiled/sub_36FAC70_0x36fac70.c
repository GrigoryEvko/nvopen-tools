// Function: sub_36FAC70
// Address: 0x36fac70
//
__int64 __fastcall sub_36FAC70(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rdi
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rdx
  unsigned int v8; // r15d
  unsigned int v9; // r14d
  __int64 v10; // rax
  unsigned __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 v15; // r15
  unsigned __int64 *v16; // rax
  char v17; // dl
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  int v21; // edx
  __int64 v22; // r9
  int v23; // ecx
  unsigned int v24; // edx
  int v25; // r8d
  int i; // r10d
  _QWORD *v27; // [rsp+10h] [rbp-160h] BYREF
  __int64 v28; // [rsp+18h] [rbp-158h]
  _QWORD v29[16]; // [rsp+20h] [rbp-150h] BYREF
  __int64 v30; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned __int64 *v31; // [rsp+A8h] [rbp-C8h]
  __int64 v32; // [rsp+B0h] [rbp-C0h]
  int v33; // [rsp+B8h] [rbp-B8h]
  char v34; // [rsp+BCh] [rbp-B4h]
  _QWORD *v35; // [rsp+C0h] [rbp-B0h] BYREF

  v3 = v29;
  v31 = (unsigned __int64 *)&v35;
  v27 = v29;
  v29[0] = a2;
  v33 = 0;
  v34 = 1;
  v35 = a2;
  v30 = 1;
  v28 = 0x1000000001LL;
  v32 = 0x100000010LL;
  v4 = 1;
  while ( 1 )
  {
    v5 = v4--;
    v6 = v3[v5 - 1];
    LODWORD(v28) = v4;
    v7 = *(_QWORD *)(v6 + 32);
    if ( *(_BYTE *)v7 )
      break;
    LOBYTE(a2) = *(_WORD *)(v6 + 68) == 0 || *(_WORD *)(v6 + 68) == 68;
    v8 = (unsigned int)a2;
    if ( (_BYTE)a2 )
    {
      v9 = (*(_DWORD *)(v6 + 40) & 0xFFFFFF) - 1;
      if ( v9 > 1 )
      {
        while ( 1 )
        {
          v10 = v7 + 40LL * (v9 - 1);
          if ( *(_BYTE *)v10 )
          {
            v3 = v27;
            v8 = 0;
            goto LABEL_20;
          }
          LODWORD(a2) = *(_DWORD *)(v10 + 8);
          v15 = sub_2EBEE10(*(_QWORD *)(a1 + 200), (int)a2);
          if ( !v34 )
            goto LABEL_13;
          v16 = v31;
          v12 = HIDWORD(v32);
          v11 = &v31[HIDWORD(v32)];
          if ( v31 != v11 )
          {
            while ( v15 != *v16 )
            {
              if ( v11 == ++v16 )
                goto LABEL_25;
            }
LABEL_11:
            v9 -= 2;
            if ( v9 <= 1 )
              goto LABEL_17;
            goto LABEL_12;
          }
LABEL_25:
          if ( HIDWORD(v32) < (unsigned int)v32 )
          {
            ++HIDWORD(v32);
            *v11 = v15;
            ++v30;
          }
          else
          {
LABEL_13:
            LODWORD(a2) = v15;
            sub_C8CC70((__int64)&v30, v15, (__int64)v11, v12, v13, v14);
            if ( !v17 )
              goto LABEL_11;
          }
          v18 = (unsigned int)v28;
          v19 = (unsigned int)v28 + 1LL;
          if ( v19 > HIDWORD(v28) )
          {
            a2 = v29;
            sub_C8D5F0((__int64)&v27, v29, v19, 8u, v13, v14);
            v18 = (unsigned int)v28;
          }
          v9 -= 2;
          v27[v18] = v15;
          LODWORD(v28) = v28 + 1;
          if ( v9 <= 1 )
          {
LABEL_17:
            v4 = v28;
            v3 = v27;
            break;
          }
LABEL_12:
          v7 = *(_QWORD *)(v6 + 32);
        }
      }
    }
    else
    {
      LODWORD(a2) = *(_DWORD *)(v7 + 8);
      v21 = *(_DWORD *)(a1 + 304);
      v22 = *(_QWORD *)(a1 + 288);
      if ( !v21 )
        goto LABEL_20;
      v23 = v21 - 1;
      v24 = (v21 - 1) & (37 * (_DWORD)a2);
      v25 = *(_DWORD *)(v22 + 4LL * v24);
      if ( (_DWORD)a2 != v25 )
      {
        for ( i = 1; ; ++i )
        {
          if ( v25 == -1 )
            goto LABEL_20;
          v24 = v23 & (i + v24);
          v25 = *(_DWORD *)(v22 + 4LL * v24);
          if ( (_DWORD)a2 == v25 )
            break;
        }
      }
    }
    if ( !v4 )
    {
      v8 = 1;
      goto LABEL_20;
    }
  }
  v8 = 0;
LABEL_20:
  if ( v3 != v29 )
    _libc_free((unsigned __int64)v3);
  if ( !v34 )
    _libc_free((unsigned __int64)v31);
  return v8;
}
