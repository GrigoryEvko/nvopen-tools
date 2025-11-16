// Function: sub_2AF8CE0
// Address: 0x2af8ce0
//
void __fastcall sub_2AF8CE0(__int64 a1)
{
  _QWORD *v1; // rcx
  bool v2; // zf
  __int64 v3; // rdx
  unsigned int v4; // eax
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // r12
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  _QWORD *v18; // rbx
  _QWORD *v19; // r14
  __int64 v20; // r8
  _QWORD *v21; // r15
  __int64 *v22; // rax
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // [rsp+18h] [rbp-168h]
  _QWORD *v26; // [rsp+20h] [rbp-160h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-158h]
  unsigned int v28; // [rsp+2Ch] [rbp-154h]
  _QWORD v29[16]; // [rsp+30h] [rbp-150h] BYREF
  __int64 v30; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 *v31; // [rsp+B8h] [rbp-C8h]
  __int64 v32; // [rsp+C0h] [rbp-C0h]
  int v33; // [rsp+C8h] [rbp-B8h]
  char v34; // [rsp+CCh] [rbp-B4h]
  char v35; // [rsp+D0h] [rbp-B0h] BYREF

  v1 = v29;
  v2 = *(_BYTE *)a1 == 84;
  v30 = 0;
  v31 = (__int64 *)&v35;
  v32 = 16;
  v33 = 0;
  v34 = 1;
  v26 = v29;
  v28 = 16;
  if ( v2 )
    return;
  v3 = *(_QWORD *)(a1 + 40);
  v4 = 1;
  v29[0] = a1;
  v27 = 1;
  while ( 1 )
  {
    v5 = v4--;
    v6 = 0;
    v7 = v1[v5 - 1];
    v27 = v4;
    v8 = 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
    if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v9 = (*(_BYTE *)(v7 + 7) & 0x40) != 0 ? *(_QWORD *)(v7 - 8) : v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
          v10 = *(_QWORD *)(v9 + v6);
          if ( *(_BYTE *)v10 > 0x1Cu && *(_BYTE *)v10 != 84 && *(_QWORD *)(v10 + 40) == v3 )
            break;
          v6 += 32;
          if ( v6 == v8 )
            goto LABEL_14;
        }
        if ( sub_B445A0(*(_QWORD *)(v9 + v6), a1) )
          goto LABEL_13;
        if ( !v34 )
          goto LABEL_26;
        v15 = v31;
        v12 = HIDWORD(v32);
        v11 = &v31[HIDWORD(v32)];
        if ( v31 != v11 )
        {
          while ( v10 != *v15 )
          {
            if ( v11 == ++v15 )
              goto LABEL_25;
          }
          goto LABEL_22;
        }
LABEL_25:
        if ( HIDWORD(v32) < (unsigned int)v32 )
        {
          ++HIDWORD(v32);
          *v11 = v10;
          ++v30;
        }
        else
        {
LABEL_26:
          sub_C8CC70((__int64)&v30, v10, (__int64)v11, v12, v13, v14);
        }
LABEL_22:
        v16 = v27;
        v17 = v27 + 1LL;
        if ( v17 > v28 )
        {
          sub_C8D5F0((__int64)&v26, v29, v17, 8u, v13, v14);
          v16 = v27;
        }
        v26[v16] = v10;
        ++v27;
LABEL_13:
        v6 += 32;
        v3 = *(_QWORD *)(a1 + 40);
        if ( v6 == v8 )
        {
LABEL_14:
          v4 = v27;
          break;
        }
      }
    }
    if ( !v4 )
      break;
    v1 = v26;
  }
  v18 = (_QWORD *)(v3 + 48);
  v19 = (_QWORD *)(a1 + 24);
  if ( a1 + 24 != v3 + 48 )
  {
    do
    {
      v20 = (__int64)(v19 - 3);
      if ( !v19 )
        v20 = 0;
      v21 = (_QWORD *)v20;
      if ( v34 )
      {
        v22 = v31;
        v23 = &v31[HIDWORD(v32)];
        if ( v31 == v23 )
          goto LABEL_36;
        while ( v20 != *v22 )
        {
          if ( v23 == ++v22 )
            goto LABEL_36;
        }
      }
      else if ( !sub_C8CA60((__int64)&v30, v20) )
      {
        goto LABEL_36;
      }
      v19 = (_QWORD *)(*v19 & 0xFFFFFFFFFFFFFFF8LL);
      sub_B43D10(v21);
      v24 = v25;
      LOWORD(v24) = 0;
      v25 = v24;
      sub_B44220(v21, a1 + 24, v24);
LABEL_36:
      v19 = (_QWORD *)v19[1];
    }
    while ( v18 != v19 );
  }
  if ( v26 != v29 )
    _libc_free((unsigned __int64)v26);
  if ( !v34 )
    _libc_free((unsigned __int64)v31);
}
