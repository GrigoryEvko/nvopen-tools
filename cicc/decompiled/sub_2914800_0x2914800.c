// Function: sub_2914800
// Address: 0x2914800
//
void __fastcall sub_2914800(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 *v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned __int8 *v17; // r15
  int v18; // edx
  __int64 v19; // r14
  _BYTE *v20; // rbx
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 *v25; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v26; // [rsp+8h] [rbp-D8h]
  _BYTE v27[32]; // [rsp+10h] [rbp-D0h] BYREF
  char *v28; // [rsp+30h] [rbp-B0h]
  __int64 v29; // [rsp+38h] [rbp-A8h]
  char v30; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v31; // [rsp+70h] [rbp-70h] BYREF
  __int64 *v32; // [rsp+78h] [rbp-68h]
  __int64 v33; // [rsp+80h] [rbp-60h]
  int v34; // [rsp+88h] [rbp-58h]
  char v35; // [rsp+8Ch] [rbp-54h]
  char v36; // [rsp+90h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  v28 = &v30;
  v29 = 0x600000000LL;
  v32 = (__int64 *)&v36;
  v26 = 0x400000000LL;
  v31 = 0;
  v33 = 4;
  v34 = 0;
  v35 = 1;
  v25 = (__int64 *)v27;
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 24);
LABEL_3:
    v8 = v32;
    v9 = HIDWORD(v33);
    v10 = &v32[HIDWORD(v33)];
    if ( v32 == v10 )
    {
LABEL_33:
      if ( HIDWORD(v33) >= (unsigned int)v33 )
        goto LABEL_9;
      ++HIDWORD(v33);
      *v10 = v7;
      ++v31;
LABEL_10:
      v11 = (unsigned int)v26;
      v9 = HIDWORD(v26);
      v12 = (unsigned int)v26 + 1LL;
      if ( v12 > HIDWORD(v26) )
      {
        sub_C8D5F0((__int64)&v25, v27, v12, 8u, a5, a6);
        v11 = (unsigned int)v26;
      }
      v10 = v25;
      v25[v11] = v7;
      LODWORD(v26) = v26 + 1;
      v6 = *(_QWORD *)(v6 + 8);
      if ( v6 )
        goto LABEL_8;
    }
    else
    {
      while ( v7 != *v8 )
      {
        if ( v10 == ++v8 )
          goto LABEL_33;
      }
      while ( 1 )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          break;
LABEL_8:
        v7 = *(_QWORD *)(v6 + 24);
        if ( v35 )
          goto LABEL_3;
LABEL_9:
        sub_C8CC70((__int64)&v31, v7, (__int64)v10, v9, a5, a6);
        if ( (_BYTE)v10 )
          goto LABEL_10;
      }
    }
    v13 = v26;
    v14 = v25;
    while ( (_DWORD)v26 )
    {
      while ( 1 )
      {
        v16 = v13--;
        v17 = (unsigned __int8 *)v14[v16 - 1];
        LODWORD(v26) = v13;
        v18 = *v17;
        if ( (_BYTE)v18 == 63 )
          break;
        v15 = (unsigned int)(v18 - 78);
        if ( (unsigned __int8)v15 <= 1u )
          goto LABEL_20;
LABEL_17:
        if ( !v13 )
          goto LABEL_29;
      }
      if ( !(unsigned __int8)sub_B4DCF0((__int64)v17) )
      {
        v13 = v26;
        v14 = v25;
        goto LABEL_17;
      }
LABEL_20:
      v19 = *((_QWORD *)v17 + 2);
      if ( v19 )
      {
        while ( 1 )
        {
          v20 = *(_BYTE **)(v19 + 24);
          if ( *v20 <= 0x1Cu )
            goto LABEL_27;
          if ( v35 )
          {
            v21 = v32;
            v9 = HIDWORD(v33);
            v15 = (__int64)&v32[HIDWORD(v33)];
            if ( v32 != (__int64 *)v15 )
            {
              while ( v20 != (_BYTE *)*v21 )
              {
                if ( (__int64 *)v15 == ++v21 )
                  goto LABEL_40;
              }
              goto LABEL_27;
            }
LABEL_40:
            if ( HIDWORD(v33) < (unsigned int)v33 )
            {
              ++HIDWORD(v33);
              *(_QWORD *)v15 = v20;
              ++v31;
              goto LABEL_36;
            }
          }
          sub_C8CC70((__int64)&v31, *(_QWORD *)(v19 + 24), v15, v9, a5, a6);
          if ( (_BYTE)v15 )
          {
LABEL_36:
            v23 = (unsigned int)v26;
            v9 = HIDWORD(v26);
            v24 = (unsigned int)v26 + 1LL;
            if ( v24 > HIDWORD(v26) )
            {
              sub_C8D5F0((__int64)&v25, v27, v24, 8u, a5, a6);
              v23 = (unsigned int)v26;
            }
            v15 = (__int64)v25;
            v25[v23] = (__int64)v20;
            LODWORD(v26) = v26 + 1;
            v19 = *(_QWORD *)(v19 + 8);
            if ( !v19 )
              break;
          }
          else
          {
LABEL_27:
            v19 = *(_QWORD *)(v19 + 8);
            if ( !v19 )
              break;
          }
        }
      }
      v22 = sub_ACADE0(*((__int64 ***)v17 + 1));
      sub_BD84D0((__int64)v17, v22);
      sub_B43D60(v17);
      v13 = v26;
      v14 = v25;
    }
LABEL_29:
    if ( v14 != (__int64 *)v27 )
      _libc_free((unsigned __int64)v14);
  }
  if ( !v35 )
    _libc_free((unsigned __int64)v32);
}
