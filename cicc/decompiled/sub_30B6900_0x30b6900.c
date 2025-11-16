// Function: sub_30B6900
// Address: 0x30b6900
//
void __fastcall sub_30B6900(__int64 a1, _BYTE *a2)
{
  bool v2; // zf
  unsigned int v3; // eax
  _QWORD *v4; // r8
  _QWORD *v5; // rcx
  __int64 v6; // rdi
  unsigned __int16 v7; // dx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // r12
  __int64 *v14; // r15
  __int64 v15; // rbx
  __int64 *v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  _BYTE *v19; // [rsp+0h] [rbp-F0h]
  _QWORD *v20; // [rsp+8h] [rbp-E8h] BYREF
  __int64 v21; // [rsp+10h] [rbp-E0h]
  _QWORD v22[8]; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v23; // [rsp+58h] [rbp-98h] BYREF
  __int64 *v24; // [rsp+60h] [rbp-90h]
  __int64 v25; // [rsp+68h] [rbp-88h]
  int v26; // [rsp+70h] [rbp-80h]
  char v27; // [rsp+74h] [rbp-7Ch]
  __int64 v28; // [rsp+78h] [rbp-78h] BYREF

  v2 = *(_WORD *)(a1 + 24) == 15;
  v21 = 0x800000000LL;
  v24 = &v28;
  v19 = a2;
  v20 = v22;
  v25 = 0x100000008LL;
  v26 = 0;
  v27 = 1;
  v28 = a1;
  v23 = 1;
  if ( v2 && (unsigned int)**(unsigned __int8 **)(a1 - 8) - 12 <= 1 )
  {
    *a2 = 1;
    v3 = 0;
  }
  else
  {
    v22[0] = a1;
    v3 = 1;
    LODWORD(v21) = 1;
  }
  v4 = v22;
  while ( 1 )
  {
    v5 = &v4[v3];
    if ( !v3 )
      break;
    while ( 1 )
    {
      if ( *a2 )
        goto LABEL_10;
      v6 = *(v5 - 1);
      LODWORD(v21) = --v3;
      v7 = *(_WORD *)(v6 + 24);
      if ( v7 > 0xEu )
      {
        if ( v7 != 15 )
          BUG();
        goto LABEL_9;
      }
      if ( v7 > 1u )
        break;
LABEL_9:
      --v5;
      if ( !v3 )
        goto LABEL_10;
    }
    v8 = sub_D960E0(v6);
    v13 = (__int64 *)(v8 + 8 * v9);
    v14 = (__int64 *)v8;
    if ( (__int64 *)v8 != v13 )
    {
      while ( 1 )
      {
        v15 = *v14;
        if ( !v27 )
          goto LABEL_26;
        v16 = v24;
        v10 = HIDWORD(v25);
        v9 = (__int64)&v24[HIDWORD(v25)];
        if ( v24 != (__int64 *)v9 )
        {
          while ( v15 != *v16 )
          {
            if ( (__int64 *)v9 == ++v16 )
              goto LABEL_30;
          }
          goto LABEL_23;
        }
LABEL_30:
        if ( HIDWORD(v25) < (unsigned int)v25 )
        {
          v10 = (unsigned int)++HIDWORD(v25);
          *(_QWORD *)v9 = v15;
          ++v23;
          if ( *(_WORD *)(v15 + 24) != 15 )
          {
LABEL_32:
            v17 = (unsigned int)v21;
            v10 = HIDWORD(v21);
            v18 = (unsigned int)v21 + 1LL;
            if ( v18 > HIDWORD(v21) )
            {
              sub_C8D5F0((__int64)&v20, v22, v18, 8u, v11, v12);
              v17 = (unsigned int)v21;
            }
            v9 = (__int64)v20;
            v20[v17] = v15;
            LODWORD(v21) = v21 + 1;
            goto LABEL_23;
          }
        }
        else
        {
LABEL_26:
          sub_C8CC70((__int64)&v23, *v14, v9, v10, v11, v12);
          if ( !(_BYTE)v9 )
            goto LABEL_23;
          if ( *(_WORD *)(v15 + 24) != 15 )
            goto LABEL_32;
        }
        if ( (unsigned int)**(unsigned __int8 **)(v15 - 8) - 12 > 1 )
          goto LABEL_32;
        *v19 = 1;
LABEL_23:
        a2 = v19;
        if ( !*v19 && v13 != ++v14 )
          continue;
        goto LABEL_25;
      }
    }
    a2 = v19;
LABEL_25:
    v4 = v20;
    v3 = v21;
  }
LABEL_10:
  if ( !v27 )
  {
    _libc_free((unsigned __int64)v24);
    v4 = v20;
  }
  if ( v4 != v22 )
    _libc_free((unsigned __int64)v4);
}
