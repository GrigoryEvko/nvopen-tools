// Function: sub_310A400
// Address: 0x310a400
//
__int64 __fastcall sub_310A400(__int64 a1)
{
  _QWORD *v1; // r8
  unsigned int i; // eax
  _QWORD *v3; // rcx
  __int64 v4; // rdi
  unsigned __int16 v5; // dx
  unsigned int v6; // r13d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // r15
  __int64 *v14; // r14
  __int64 v15; // rbx
  __int64 *v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  int v19; // [rsp+Ch] [rbp-F4h] BYREF
  int *v20; // [rsp+10h] [rbp-F0h]
  _QWORD *v21; // [rsp+18h] [rbp-E8h] BYREF
  unsigned int v22; // [rsp+20h] [rbp-E0h]
  unsigned int v23; // [rsp+24h] [rbp-DCh]
  _QWORD v24[8]; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v25; // [rsp+68h] [rbp-98h] BYREF
  __int64 *v26; // [rsp+70h] [rbp-90h]
  __int64 v27; // [rsp+78h] [rbp-88h]
  int v28; // [rsp+80h] [rbp-80h]
  char v29; // [rsp+84h] [rbp-7Ch]
  __int64 v30; // [rsp+88h] [rbp-78h] BYREF

  v1 = v24;
  v20 = &v19;
  v21 = v24;
  v23 = 8;
  v28 = 0;
  v29 = 1;
  v30 = a1;
  v25 = 1;
  v19 = 1;
  v24[0] = a1;
  v22 = 1;
  v26 = &v30;
  v27 = 0x100000008LL;
  for ( i = 1; ; i = v22 )
  {
    v3 = &v1[i];
    if ( !i )
      break;
    while ( 1 )
    {
      v4 = *(v3 - 1);
      v22 = --i;
      v5 = *(_WORD *)(v4 + 24);
      if ( v5 > 0xEu )
      {
        if ( v5 != 15 )
          BUG();
        goto LABEL_5;
      }
      if ( v5 > 1u )
        break;
LABEL_5:
      --v3;
      if ( !i )
        goto LABEL_6;
    }
    v8 = sub_D960E0(v4);
    v13 = (__int64 *)(v8 + 8 * v9);
    v14 = (__int64 *)v8;
    if ( (__int64 *)v8 != v13 )
    {
      while ( 1 )
      {
        v15 = *v14;
        if ( v29 )
        {
          v16 = v26;
          v10 = HIDWORD(v27);
          v9 = (__int64)&v26[HIDWORD(v27)];
          if ( v26 != (__int64 *)v9 )
          {
            while ( v15 != *v16 )
            {
              if ( (__int64 *)v9 == ++v16 )
                goto LABEL_26;
            }
            goto LABEL_19;
          }
LABEL_26:
          if ( HIDWORD(v27) < (unsigned int)v27 )
          {
            ++HIDWORD(v27);
            *(_QWORD *)v9 = v15;
            ++v25;
            goto LABEL_22;
          }
        }
        sub_C8CC70((__int64)&v25, *v14, v9, v10, v11, v12);
        if ( (_BYTE)v9 )
        {
LABEL_22:
          ++*v20;
          v17 = v22;
          v10 = v23;
          v18 = v22 + 1LL;
          if ( v18 > v23 )
          {
            sub_C8D5F0((__int64)&v21, v24, v18, 8u, v11, v12);
            v17 = v22;
          }
          v9 = (__int64)v21;
          ++v14;
          v21[v17] = v15;
          ++v22;
          if ( v13 == v14 )
            break;
        }
        else
        {
LABEL_19:
          if ( v13 == ++v14 )
            break;
        }
      }
    }
    v1 = v21;
  }
LABEL_6:
  v6 = v19;
  if ( !v29 )
  {
    _libc_free((unsigned __int64)v26);
    v1 = v21;
  }
  if ( v1 != v24 )
    _libc_free((unsigned __int64)v1);
  return v6;
}
