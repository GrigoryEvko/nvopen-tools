// Function: sub_30B66A0
// Address: 0x30b66a0
//
void __fastcall sub_30B66A0(__int64 a1, _BYTE **a2)
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
  __int64 *v13; // r15
  __int64 *v14; // r14
  __int64 v15; // rbx
  __int64 *v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  _QWORD *v19; // [rsp+8h] [rbp-E8h] BYREF
  __int64 v20; // [rsp+10h] [rbp-E0h]
  _QWORD v21[8]; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v22; // [rsp+58h] [rbp-98h] BYREF
  __int64 *v23; // [rsp+60h] [rbp-90h]
  __int64 v24; // [rsp+68h] [rbp-88h]
  int v25; // [rsp+70h] [rbp-80h]
  char v26; // [rsp+74h] [rbp-7Ch]
  __int64 v27; // [rsp+78h] [rbp-78h] BYREF

  v2 = *(_WORD *)(a1 + 24) == 8;
  v20 = 0x800000000LL;
  v23 = &v27;
  v19 = v21;
  v24 = 0x100000008LL;
  v25 = 0;
  v26 = 1;
  v27 = a1;
  v22 = 1;
  if ( v2 )
  {
    **a2 = 1;
    v4 = v19;
    v3 = v20;
  }
  else
  {
    v21[0] = a1;
    v3 = 1;
    v4 = v21;
    LODWORD(v20) = 1;
  }
  while ( 1 )
  {
    v5 = &v4[v3];
    if ( !v3 )
      break;
    while ( 1 )
    {
      v6 = *(v5 - 1);
      LODWORD(v20) = --v3;
      v7 = *(_WORD *)(v6 + 24);
      if ( v7 > 0xEu )
      {
        if ( v7 != 15 )
          BUG();
        goto LABEL_6;
      }
      if ( v7 > 1u )
        break;
LABEL_6:
      --v5;
      if ( !v3 )
        goto LABEL_7;
    }
    v8 = sub_D960E0(v6);
    v13 = (__int64 *)(v8 + 8 * v9);
    v14 = (__int64 *)v8;
    if ( (__int64 *)v8 != v13 )
    {
      while ( 1 )
      {
        v15 = *v14;
        if ( !v26 )
          goto LABEL_22;
        v16 = v23;
        v10 = HIDWORD(v24);
        v9 = (__int64)&v23[HIDWORD(v24)];
        if ( v23 != (__int64 *)v9 )
        {
          while ( v15 != *v16 )
          {
            if ( (__int64 *)v9 == ++v16 )
              goto LABEL_28;
          }
          goto LABEL_20;
        }
LABEL_28:
        if ( HIDWORD(v24) < (unsigned int)v24 )
        {
          v10 = (unsigned int)++HIDWORD(v24);
          *(_QWORD *)v9 = v15;
          ++v22;
          if ( *(_WORD *)(v15 + 24) == 8 )
            goto LABEL_30;
LABEL_24:
          v17 = (unsigned int)v20;
          v10 = HIDWORD(v20);
          v18 = (unsigned int)v20 + 1LL;
          if ( v18 > HIDWORD(v20) )
          {
            sub_C8D5F0((__int64)&v19, v21, v18, 8u, v11, v12);
            v17 = (unsigned int)v20;
          }
          v9 = (__int64)v19;
          ++v14;
          v19[v17] = v15;
          LODWORD(v20) = v20 + 1;
          if ( v13 == v14 )
            break;
        }
        else
        {
LABEL_22:
          sub_C8CC70((__int64)&v22, *v14, v9, v10, v11, v12);
          if ( (_BYTE)v9 )
          {
            if ( *(_WORD *)(v15 + 24) != 8 )
              goto LABEL_24;
LABEL_30:
            ++v14;
            **a2 = 1;
            if ( v13 == v14 )
              break;
          }
          else
          {
LABEL_20:
            if ( v13 == ++v14 )
              break;
          }
        }
      }
    }
    v4 = v19;
    v3 = v20;
  }
LABEL_7:
  if ( !v26 )
  {
    _libc_free((unsigned __int64)v23);
    v4 = v19;
  }
  if ( v4 != v21 )
    _libc_free((unsigned __int64)v4);
}
