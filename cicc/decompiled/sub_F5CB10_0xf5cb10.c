// Function: sub_F5CB10
// Address: 0xf5cb10
//
__int64 __fastcall sub_F5CB10(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v3; // r14
  __int64 v5; // rax
  char v6; // r12
  __int64 v7; // rdx
  __int64 *v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // r15d
  __int64 *v13; // rax
  __int64 v14; // rax
  char v15; // dl
  __int64 *v17; // [rsp+8h] [rbp-98h]
  _BYTE v18[16]; // [rsp+10h] [rbp-90h] BYREF
  void (__fastcall *v19)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-80h]
  __int64 v20; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v21; // [rsp+38h] [rbp-68h]
  __int64 v22; // [rsp+40h] [rbp-60h]
  int v23; // [rsp+48h] [rbp-58h]
  char v24; // [rsp+4Ch] [rbp-54h]
  char v25; // [rsp+50h] [rbp-50h] BYREF

  v3 = a1;
  v21 = (__int64 *)&v25;
  v5 = *(_QWORD *)(a1 + 16);
  v17 = a2;
  v20 = 0;
  v22 = 4;
  v23 = 0;
  v24 = 1;
  if ( v5 )
  {
    v6 = 1;
    while ( 2 )
    {
      v7 = *(_QWORD *)(v5 + 24);
      while ( 1 )
      {
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          break;
        if ( v7 != *(_QWORD *)(v5 + 24) )
          goto LABEL_21;
      }
      v12 = sub_B46970((unsigned __int8 *)v3);
      if ( (_BYTE)v12 )
      {
LABEL_21:
        v12 = 0;
        goto LABEL_18;
      }
      if ( v6 )
      {
        v13 = v21;
        v9 = HIDWORD(v22);
        v8 = &v21[HIDWORD(v22)];
        if ( v21 != v8 )
        {
          while ( *v13 != v3 )
          {
            if ( v8 == ++v13 )
              goto LABEL_22;
          }
          goto LABEL_12;
        }
LABEL_22:
        if ( HIDWORD(v22) < (unsigned int)v22 )
        {
          ++HIDWORD(v22);
          *v8 = v3;
          v6 = v24;
          ++v20;
LABEL_16:
          v3 = *(_QWORD *)(*(_QWORD *)(v3 + 16) + 24LL);
          v5 = *(_QWORD *)(v3 + 16);
          if ( v5 )
            continue;
          if ( (unsigned __int8)sub_B46970((unsigned __int8 *)v3) )
            goto LABEL_18;
          goto LABEL_25;
        }
      }
      break;
    }
    a2 = (__int64 *)v3;
    sub_C8CC70((__int64)&v20, v3, (__int64)v8, v9, v10, v11);
    v6 = v24;
    if ( v15 )
      goto LABEL_16;
LABEL_12:
    v14 = sub_ACADE0(*(__int64 ***)(v3 + 8));
    sub_BD84D0(v3, v14);
    a2 = v17;
    v19 = 0;
    sub_F5CAB0((char *)v3, v17, a3, (__int64)v18);
    if ( v19 )
    {
      a2 = (__int64 *)v18;
      v19(v18, v18, 3);
    }
    v6 = v24;
    v12 = 1;
    goto LABEL_18;
  }
  v12 = 0;
  if ( (unsigned __int8)sub_B46970((unsigned __int8 *)a1) )
    return v12;
LABEL_25:
  a2 = v17;
  v19 = 0;
  v12 = sub_F5CAB0((char *)v3, v17, a3, (__int64)v18);
  if ( v19 )
  {
    a2 = (__int64 *)v18;
    v19(v18, v18, 3);
  }
  v6 = v24;
LABEL_18:
  if ( !v6 )
    _libc_free(v21, a2);
  return v12;
}
