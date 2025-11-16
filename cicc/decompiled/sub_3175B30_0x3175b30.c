// Function: sub_3175B30
// Address: 0x3175b30
//
__int64 __fastcall sub_3175B30(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  unsigned __int8 *v6; // r12
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rax
  char *v14; // rdi
  char *v15; // rcx
  char v16; // al
  unsigned int v17; // eax
  unsigned int v18; // edx
  bool v19; // al
  unsigned int v20; // r12d
  const void *v21; // rax
  unsigned int v22; // eax
  unsigned __int8 v23; // al
  bool v24; // al
  unsigned int v25; // [rsp+0h] [rbp-A0h]
  __int64 v26; // [rsp+0h] [rbp-A0h]
  unsigned int v27; // [rsp+0h] [rbp-A0h]
  _BYTE *v28; // [rsp+8h] [rbp-98h]
  __int64 v29; // [rsp+8h] [rbp-98h]
  __int64 v30; // [rsp+8h] [rbp-98h]
  __int64 v31; // [rsp+8h] [rbp-98h]
  __int64 v32; // [rsp+8h] [rbp-98h]
  __int64 v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+8h] [rbp-98h]
  __int64 v35; // [rsp+8h] [rbp-98h]
  __int64 v36; // [rsp+8h] [rbp-98h]
  __int64 v37; // [rsp+8h] [rbp-98h]
  __int64 v38; // [rsp+8h] [rbp-98h]
  unsigned __int64 v39; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v40; // [rsp+18h] [rbp-88h]
  const void *v41; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v42; // [rsp+28h] [rbp-78h]
  const void *v43; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v44; // [rsp+38h] [rbp-68h]
  __int16 v45; // [rsp+40h] [rbp-60h] BYREF
  const void *v46; // [rsp+48h] [rbp-58h] BYREF
  unsigned int v47; // [rsp+50h] [rbp-50h]
  const void *v48; // [rsp+58h] [rbp-48h] BYREF
  unsigned int v49; // [rsp+60h] [rbp-40h]

  v3 = (__int64 *)a1[30];
  v4 = *(_QWORD *)(a2 - 32);
  v5 = *v3;
  v6 = (unsigned __int8 *)v3[1];
  if ( *v3 == v4 )
  {
    v28 = *(_BYTE **)(a2 - 64);
    v11 = sub_31751A0((__int64)a1, v28);
    v8 = (__int64)v28;
    if ( v11 )
    {
      v9 = (__int64)v6;
      v6 = (unsigned __int8 *)v11;
      return sub_9719A0(*(_WORD *)(a2 + 2) & 0x3F, v6, v9, a1[5], 0, 0);
    }
  }
  else
  {
    v7 = sub_31751A0((__int64)a1, *(_BYTE **)(a2 - 32));
    v8 = v4;
    v9 = v7;
    if ( v7 )
      return sub_9719A0(*(_WORD *)(a2 + 2) & 0x3F, v6, v9, a1[5], 0, 0);
  }
  v45 = 0;
  v12 = *v6;
  if ( (unsigned int)(v12 - 12) > 1 )
  {
    if ( (_BYTE)v12 != 17 )
    {
      LOBYTE(v45) = 2;
      v46 = v6;
      goto LABEL_8;
    }
    v40 = *((_DWORD *)v6 + 8);
    if ( v40 > 0x40 )
    {
      v26 = v8;
      sub_C43780((__int64)&v39, (const void **)v6 + 3);
      v8 = v26;
    }
    else
    {
      v39 = *((_QWORD *)v6 + 3);
    }
    v29 = v8;
    sub_AADBC0((__int64)&v41, (__int64 *)&v39);
    v16 = sub_AAF760((__int64)&v41);
    v8 = v29;
    if ( v16 )
    {
      if ( (_BYTE)v45 != 6 )
      {
        if ( (unsigned int)(unsigned __int8)v45 - 4 <= 1 )
        {
          if ( v49 > 0x40 && v48 )
          {
            j_j___libc_free_0_0((unsigned __int64)v48);
            v8 = v29;
          }
          if ( v47 > 0x40 && v46 )
          {
            v38 = v8;
            j_j___libc_free_0_0((unsigned __int64)v46);
            v8 = v38;
          }
        }
        LOBYTE(v45) = 6;
      }
      goto LABEL_19;
    }
    if ( (_BYTE)v45 == 1 )
    {
      v23 = 5;
    }
    else
    {
      if ( (_BYTE)v45 == 5 || (_BYTE)v45 == 4 )
      {
        v18 = v47;
        if ( v47 <= 0x40 )
        {
          v21 = v41;
          if ( v46 != v41 )
          {
            v20 = v49;
LABEL_43:
            v46 = v21;
            v22 = v42;
            v42 = 0;
            v47 = v22;
            if ( v20 > 0x40 && v48 )
            {
              v36 = v8;
              j_j___libc_free_0_0((unsigned __int64)v48);
              v17 = v42;
              v8 = v36;
              v48 = v43;
              v49 = v44;
LABEL_23:
              if ( v17 > 0x40 && v41 )
              {
                v31 = v8;
                j_j___libc_free_0_0((unsigned __int64)v41);
                v8 = v31;
              }
              goto LABEL_26;
            }
            v48 = v43;
            v49 = v44;
LABEL_26:
            if ( v40 > 0x40 && v39 )
            {
              v32 = v8;
              j_j___libc_free_0_0(v39);
              v8 = v32;
            }
            goto LABEL_8;
          }
        }
        else
        {
          v25 = v47;
          v19 = sub_C43C50((__int64)&v46, &v41);
          v8 = v29;
          v18 = v25;
          if ( !v19 )
            goto LABEL_41;
        }
        v20 = v49;
        if ( v49 <= 0x40 )
        {
          if ( v48 != v43 )
          {
LABEL_51:
            if ( v18 <= 0x40 )
            {
LABEL_52:
              v21 = v41;
              goto LABEL_43;
            }
LABEL_41:
            if ( v46 )
            {
              v35 = v8;
              j_j___libc_free_0_0((unsigned __int64)v46);
              v20 = v49;
              v21 = v41;
              v8 = v35;
              goto LABEL_43;
            }
            v20 = v49;
            goto LABEL_52;
          }
        }
        else
        {
          v27 = v18;
          v37 = v8;
          v24 = sub_C43C50((__int64)&v48, &v43);
          v8 = v37;
          v18 = v27;
          if ( !v24 )
            goto LABEL_51;
        }
LABEL_19:
        if ( v44 > 0x40 && v43 )
        {
          v30 = v8;
          j_j___libc_free_0_0((unsigned __int64)v43);
          v8 = v30;
        }
        v17 = v42;
        goto LABEL_23;
      }
      v23 = 4;
    }
    v45 = v23;
    v47 = v42;
    v46 = v41;
    v49 = v44;
    v48 = v43;
    goto LABEL_26;
  }
  LOBYTE(v45) = 1;
LABEL_8:
  v13 = sub_2A64F10(a1[7], v8);
  v14 = (char *)&v45;
  v15 = (char *)v13;
  if ( v5 == v4 )
  {
    v14 = (char *)v13;
    v15 = (char *)&v45;
  }
  result = sub_22EAB60(v14, *(_WORD *)(a2 + 2) & 0x3F, *(_QWORD *)(a2 + 8), v15, a1[5]);
  if ( (unsigned int)(unsigned __int8)v45 - 4 <= 1 )
  {
    if ( v49 > 0x40 && v48 )
    {
      v33 = result;
      j_j___libc_free_0_0((unsigned __int64)v48);
      result = v33;
    }
    if ( v47 > 0x40 )
    {
      if ( v46 )
      {
        v34 = result;
        j_j___libc_free_0_0((unsigned __int64)v46);
        return v34;
      }
    }
  }
  return result;
}
