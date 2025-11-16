// Function: sub_1471300
// Address: 0x1471300
//
__int64 __fastcall sub_1471300(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r15d
  __int64 v6; // rbx
  _QWORD *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 *v13; // rax
  char v14; // dl
  __int64 v15; // rax
  __int64 *v16; // rsi
  __int64 *v17; // rcx
  unsigned int v18; // eax
  __int64 v19; // [rsp+10h] [rbp-140h]
  _QWORD *v21; // [rsp+20h] [rbp-130h] BYREF
  __int64 v22; // [rsp+28h] [rbp-128h]
  _QWORD v23[8]; // [rsp+30h] [rbp-120h] BYREF
  __int64 v24; // [rsp+70h] [rbp-E0h] BYREF
  __int64 *v25; // [rsp+78h] [rbp-D8h]
  __int64 *v26; // [rsp+80h] [rbp-D0h]
  __int64 v27; // [rsp+88h] [rbp-C8h]
  int v28; // [rsp+90h] [rbp-C0h]
  _BYTE v29[184]; // [rsp+98h] [rbp-B8h] BYREF

  v4 = sub_1471070(a1, a2);
  if ( (_BYTE)v4 )
    return v4;
  v19 = sub_13F9E70(a3);
  v6 = sub_13FCB50(a3);
  if ( v19 != v6 || v6 == 0 || v19 == 0 )
    return v4;
  v24 = 0;
  v25 = (__int64 *)v29;
  v26 = (__int64 *)v29;
  v21 = v23;
  v27 = 16;
  v28 = 0;
  v22 = 0x800000001LL;
  sub_1412190((__int64)&v24, a2);
  v23[0] = a2;
  v7 = v23;
  v8 = 1;
  while ( 1 )
  {
    v9 = (unsigned int)v8;
    LODWORD(v8) = v8 - 1;
    v10 = v7[v9 - 1];
    LODWORD(v22) = v8;
    v11 = *(_QWORD *)(v10 + 8);
    if ( v11 )
      break;
LABEL_15:
    if ( !(_DWORD)v8 )
    {
      v4 = 0;
      goto LABEL_28;
    }
  }
  while ( 1 )
  {
    v12 = sub_1648700(v11);
    if ( !(unsigned __int8)sub_14AEA40(v12) )
      break;
    v13 = v25;
    if ( v26 == v25 )
    {
      v16 = &v25[HIDWORD(v27)];
      if ( v25 != v16 )
      {
        v17 = 0;
        while ( v12 != *v13 )
        {
          if ( *v13 == -2 )
            v17 = v13;
          if ( v16 == ++v13 )
          {
            if ( !v17 )
              goto LABEL_32;
            *v17 = v12;
            v15 = (unsigned int)v22;
            --v28;
            ++v24;
            if ( (unsigned int)v22 < HIDWORD(v22) )
              goto LABEL_13;
            goto LABEL_25;
          }
        }
        goto LABEL_8;
      }
LABEL_32:
      if ( HIDWORD(v27) < (unsigned int)v27 )
      {
        ++HIDWORD(v27);
        *v16 = v12;
        ++v24;
        goto LABEL_12;
      }
    }
    sub_16CCBA0(&v24, v12);
    if ( v14 )
    {
LABEL_12:
      v15 = (unsigned int)v22;
      if ( (unsigned int)v22 >= HIDWORD(v22) )
      {
LABEL_25:
        sub_16CD150(&v21, v23, 0, 8);
        v15 = (unsigned int)v22;
      }
LABEL_13:
      v21[v15] = v12;
      LODWORD(v22) = v22 + 1;
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
      {
LABEL_14:
        LODWORD(v8) = v22;
        v7 = v21;
        goto LABEL_15;
      }
    }
    else
    {
LABEL_8:
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        goto LABEL_14;
    }
  }
  if ( *(_BYTE *)(v12 + 16) != 26 || v6 != *(_QWORD *)(v12 + 40) )
    goto LABEL_8;
  v18 = sub_14691E0(a1, a3);
  v7 = v21;
  v4 = v18;
LABEL_28:
  if ( v7 != v23 )
    _libc_free((unsigned __int64)v7);
  if ( v26 != v25 )
    _libc_free((unsigned __int64)v26);
  return v4;
}
