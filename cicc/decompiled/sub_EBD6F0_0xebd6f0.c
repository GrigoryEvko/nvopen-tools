// Function: sub_EBD6F0
// Address: 0xebd6f0
//
__int64 __fastcall sub_EBD6F0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int8 v2; // al
  unsigned int v3; // r12d
  __int64 *v4; // r14
  __int64 *v5; // r15
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rdi
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // r15
  _QWORD *v20; // rsi
  __int64 *v21; // [rsp+8h] [rbp-208h]
  __int64 v22; // [rsp+10h] [rbp-200h]
  __int64 *v24; // [rsp+30h] [rbp-1E0h] BYREF
  __int64 *v25; // [rsp+38h] [rbp-1D8h]
  __int64 v26; // [rsp+40h] [rbp-1D0h]
  __int64 v27[2]; // [rsp+50h] [rbp-1C0h] BYREF
  __int64 v28; // [rsp+60h] [rbp-1B0h]
  __int64 v29; // [rsp+68h] [rbp-1A8h]
  __int64 v30; // [rsp+70h] [rbp-1A0h]
  __int16 v31; // [rsp+78h] [rbp-198h]
  _QWORD v32[4]; // [rsp+80h] [rbp-190h] BYREF
  __int64 v33; // [rsp+A0h] [rbp-170h]
  __int64 v34; // [rsp+A8h] [rbp-168h]
  _QWORD *v35; // [rsp+B0h] [rbp-160h]
  _QWORD v36[3]; // [rsp+C0h] [rbp-150h] BYREF
  _BYTE v37[312]; // [rsp+D8h] [rbp-138h] BYREF

  v27[0] = 0;
  v27[1] = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v32[0] = "expected identifier in '.irp' directive";
  LOWORD(v33) = 259;
  v2 = sub_EB61F0(a1, v27);
  if ( (unsigned __int8)sub_ECE0A0(a1, v2, v32)
    || (v37[9] = 1, v36[0] = "expected comma", v37[8] = 3, (unsigned __int8)sub_ECE210(a1, 26, v36))
    || (unsigned __int8)sub_EBC8F0(a1, 0, &v24, v13, v14, v15)
    || (unsigned __int8)sub_ECE000(a1) )
  {
    v3 = 1;
  }
  else
  {
    v3 = 1;
    v22 = sub_EB4420(a1, a2);
    if ( v22 )
    {
      v36[0] = v37;
      v34 = 0x100000000LL;
      v35 = v36;
      v36[1] = 0;
      v32[0] = &unk_49DD288;
      v36[2] = 256;
      v32[1] = 2;
      v32[2] = 0;
      v32[3] = 0;
      v33 = 0;
      sub_CB5980((__int64)v32, 0, 0, 0);
      v19 = v24;
      v21 = v25;
      if ( v25 == v24 )
      {
LABEL_35:
        v20 = (_QWORD *)a2;
        v3 = 0;
        sub_EB41F0(a1, a2, v32, v16, v17, v18);
      }
      else
      {
        while ( 1 )
        {
          v20 = v32;
          v3 = sub_EA4200(a1, (int *)v32, v22, (__int64)v27, 1, 1, (__int64)v19, 1u);
          if ( (_BYTE)v3 )
            break;
          v19 += 3;
          if ( v21 == v19 )
            goto LABEL_35;
        }
      }
      v32[0] = &unk_49DD388;
      sub_CB5840((__int64)v32);
      if ( (_BYTE *)v36[0] != v37 )
        _libc_free(v36[0], v20);
    }
  }
  v4 = v25;
  v5 = v24;
  if ( v25 != v24 )
  {
    do
    {
      v6 = v5[1];
      v7 = *v5;
      if ( v6 != *v5 )
      {
        do
        {
          if ( *(_DWORD *)(v7 + 32) > 0x40u )
          {
            v8 = *(_QWORD *)(v7 + 24);
            if ( v8 )
              j_j___libc_free_0_0(v8);
          }
          v7 += 40;
        }
        while ( v6 != v7 );
        v7 = *v5;
      }
      if ( v7 )
        j_j___libc_free_0(v7, v5[2] - v7);
      v5 += 3;
    }
    while ( v4 != v5 );
    v5 = v24;
  }
  if ( v5 )
    j_j___libc_free_0(v5, v26 - (_QWORD)v5);
  v9 = v29;
  v10 = v28;
  if ( v29 != v28 )
  {
    do
    {
      if ( *(_DWORD *)(v10 + 32) > 0x40u )
      {
        v11 = *(_QWORD *)(v10 + 24);
        if ( v11 )
          j_j___libc_free_0_0(v11);
      }
      v10 += 40;
    }
    while ( v9 != v10 );
    v10 = v28;
  }
  if ( v10 )
    j_j___libc_free_0(v10, v30 - v10);
  return v3;
}
