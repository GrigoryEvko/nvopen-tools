// Function: sub_1000660
// Address: 0x1000660
//
__int64 __fastcall sub_1000660(int a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5, unsigned __int8 *a6)
{
  __int64 v6; // r13
  __int64 v11; // rsi
  unsigned __int8 *v12; // r9
  __int64 *v13; // rax
  __int64 *v14; // r15
  char v15; // cl
  __int64 *v16; // r14
  char *v18; // rax
  char *v19; // rdx
  __int64 v20; // rdx
  _QWORD *v21; // rdi
  int v22; // ecx
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 *v25; // rax
  __int64 v26; // [rsp+8h] [rbp-B8h]
  __int64 v27; // [rsp+10h] [rbp-B0h] BYREF
  __int64 *v28; // [rsp+18h] [rbp-A8h]
  __int64 v29; // [rsp+20h] [rbp-A0h]
  int v30; // [rsp+28h] [rbp-98h]
  char v31; // [rsp+2Ch] [rbp-94h]
  char v32; // [rsp+30h] [rbp-90h] BYREF
  __int64 v33; // [rsp+50h] [rbp-70h] BYREF
  char *v34; // [rsp+58h] [rbp-68h]
  __int64 v35; // [rsp+60h] [rbp-60h]
  int v36; // [rsp+68h] [rbp-58h]
  char v37; // [rsp+6Ch] [rbp-54h]
  char v38; // [rsp+70h] [rbp-50h] BYREF

  v6 = 0;
  if ( (unsigned int)(a1 - 35) <= 1 )
  {
    v28 = (__int64 *)&v32;
    v34 = &v38;
    v27 = 0;
    v29 = 4;
    v30 = 0;
    v31 = 1;
    v33 = 0;
    v35 = 4;
    v36 = 0;
    v37 = 1;
    sub_FFF9B0((__int64)&v27, (unsigned __int8 *)a2, 0, a4, 0, a6);
    v11 = (__int64)a3;
    sub_FFF9B0((__int64)&v33, a3, 1, a4, 0, v12);
    v13 = v28;
    if ( v31 )
      v14 = &v28[HIDWORD(v29)];
    else
      v14 = &v28[(unsigned int)v29];
    v15 = v37;
    if ( v28 != v14 )
    {
      while ( 1 )
      {
        v11 = *v13;
        v16 = v13;
        if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v14 == ++v13 )
          goto LABEL_7;
      }
      while ( v14 != v16 )
      {
        if ( v15 )
        {
          v18 = v34;
          v19 = &v34[8 * HIDWORD(v35)];
          if ( v34 != v19 )
          {
            while ( v11 != *(_QWORD *)v18 )
            {
              v18 += 8;
              if ( v19 == v18 )
                goto LABEL_25;
            }
LABEL_17:
            v20 = *(_QWORD *)(a2 + 8);
            v21 = *(_QWORD **)v20;
            v22 = *(unsigned __int8 *)(v20 + 8);
            if ( (unsigned int)(v22 - 17) > 1 )
            {
              v24 = sub_BCB2A0(v21);
            }
            else
            {
              BYTE4(v26) = (_BYTE)v22 == 18;
              LODWORD(v26) = *(_DWORD *)(v20 + 32);
              v23 = (__int64 *)sub_BCB2A0(v21);
              v24 = sub_BCE1B0(v23, v26);
            }
            v11 = a1 == 35;
            v6 = sub_AD64A0(v24, v11);
            if ( !v37 )
              goto LABEL_9;
            goto LABEL_20;
          }
        }
        else
        {
          if ( sub_C8CA60((__int64)&v33, v11) )
            goto LABEL_17;
          v15 = v37;
        }
LABEL_25:
        v25 = v16 + 1;
        if ( v16 + 1 == v14 )
          break;
        while ( 1 )
        {
          v11 = *v25;
          v16 = v25;
          if ( (unsigned __int64)*v25 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v14 == ++v25 )
          {
            v6 = 0;
            goto LABEL_8;
          }
        }
      }
    }
LABEL_7:
    v6 = 0;
LABEL_8:
    if ( v15 )
    {
LABEL_20:
      if ( v31 )
        return v6;
    }
    else
    {
LABEL_9:
      _libc_free(v34, v11);
      if ( v31 )
        return v6;
    }
    _libc_free(v28, v11);
  }
  return v6;
}
