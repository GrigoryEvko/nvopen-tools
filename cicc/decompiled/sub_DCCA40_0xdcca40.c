// Function: sub_DCCA40
// Address: 0xdcca40
//
__int64 __fastcall sub_DCCA40(__int64 *a1, int a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // eax
  unsigned int v7; // r12d
  __int64 v8; // r14
  unsigned int v9; // edx
  __int64 v10; // r13
  unsigned int v11; // eax
  const void *v12; // rdi
  __int64 v14; // rbx
  __int64 v15; // rbx
  unsigned int v16; // eax
  __int64 v17; // rbx
  __int64 v18; // rbx
  unsigned int v19; // eax
  __int64 v20; // r14
  _QWORD *v21; // r13
  const void *v22; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v23; // [rsp+18h] [rbp-A8h]
  __int64 v24; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v25; // [rsp+28h] [rbp-98h]
  const void *v26; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v27; // [rsp+38h] [rbp-88h]
  __int64 v28; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v29; // [rsp+48h] [rbp-78h]
  const void *v30; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v31; // [rsp+58h] [rbp-68h]
  __int64 v32; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v33; // [rsp+68h] [rbp-58h]
  const void *v34; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v35; // [rsp+78h] [rbp-48h]
  __int64 v36; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v37; // [rsp+88h] [rbp-38h]

  LOBYTE(v6) = sub_D90F00(a3, a4);
  if ( !(_BYTE)v6 )
  {
    v7 = v6;
    if ( a2 == 32 )
      return v7;
    if ( a2 == 33 )
    {
      v14 = sub_DBB9F0((__int64)a1, a3, 1u, 0);
      v23 = *(_DWORD *)(v14 + 8);
      if ( v23 > 0x40 )
        sub_C43780((__int64)&v22, (const void **)v14);
      else
        v22 = *(const void **)v14;
      v25 = *(_DWORD *)(v14 + 24);
      if ( v25 > 0x40 )
        sub_C43780((__int64)&v24, (const void **)(v14 + 16));
      else
        v24 = *(_QWORD *)(v14 + 16);
      v15 = sub_DBB9F0((__int64)a1, a4, 1u, 0);
      v27 = *(_DWORD *)(v15 + 8);
      if ( v27 > 0x40 )
        sub_C43780((__int64)&v26, (const void **)v15);
      else
        v26 = *(const void **)v15;
      v29 = *(_DWORD *)(v15 + 24);
      if ( v29 > 0x40 )
        sub_C43780((__int64)&v28, (const void **)(v15 + 16));
      else
        v28 = *(_QWORD *)(v15 + 16);
      LOBYTE(v16) = sub_ABB410((__int64 *)&v22, 33, (__int64 *)&v26);
      v7 = v16;
      if ( !(_BYTE)v16 )
      {
        v17 = sub_DBB9F0((__int64)a1, a3, 0, 0);
        v31 = *(_DWORD *)(v17 + 8);
        if ( v31 > 0x40 )
          sub_C43780((__int64)&v30, (const void **)v17);
        else
          v30 = *(const void **)v17;
        v33 = *(_DWORD *)(v17 + 24);
        if ( v33 > 0x40 )
          sub_C43780((__int64)&v32, (const void **)(v17 + 16));
        else
          v32 = *(_QWORD *)(v17 + 16);
        v18 = sub_DBB9F0((__int64)a1, a4, 0, 0);
        v35 = *(_DWORD *)(v18 + 8);
        if ( v35 > 0x40 )
          sub_C43780((__int64)&v34, (const void **)v18);
        else
          v34 = *(const void **)v18;
        v37 = *(_DWORD *)(v18 + 24);
        if ( v37 > 0x40 )
          sub_C43780((__int64)&v36, (const void **)(v18 + 16));
        else
          v36 = *(_QWORD *)(v18 + 16);
        LOBYTE(v19) = sub_ABB410((__int64 *)&v30, 33, (__int64 *)&v34);
        v7 = v19;
        if ( !(_BYTE)v19 )
        {
          v21 = sub_DCC810(a1, a3, a4, 0, 0);
          if ( !sub_D96A50((__int64)v21) )
            v7 = sub_DBE090((__int64)a1, (__int64)v21);
        }
        if ( v37 > 0x40 && v36 )
          j_j___libc_free_0_0(v36);
        if ( v35 > 0x40 && v34 )
          j_j___libc_free_0_0(v34);
        if ( v33 > 0x40 && v32 )
          j_j___libc_free_0_0(v32);
        if ( v31 > 0x40 && v30 )
          j_j___libc_free_0_0(v30);
      }
      if ( v29 > 0x40 && v28 )
        j_j___libc_free_0_0(v28);
      if ( v27 > 0x40 && v26 )
        j_j___libc_free_0_0(v26);
      if ( v25 > 0x40 && v24 )
        j_j___libc_free_0_0(v24);
      if ( v23 <= 0x40 )
        return v7;
      v12 = v22;
      if ( !v22 )
        return v7;
LABEL_24:
      j_j___libc_free_0_0(v12);
      return v7;
    }
    if ( sub_B532B0(a2) )
    {
      v20 = sub_DBB9F0((__int64)a1, a3, 1u, 0);
      v31 = *(_DWORD *)(v20 + 8);
      if ( v31 > 0x40 )
        sub_C43780((__int64)&v30, (const void **)v20);
      else
        v30 = *(const void **)v20;
      v33 = *(_DWORD *)(v20 + 24);
      if ( v33 > 0x40 )
        sub_C43780((__int64)&v32, (const void **)(v20 + 16));
      else
        v32 = *(_QWORD *)(v20 + 16);
      v9 = 1;
    }
    else
    {
      v8 = sub_DBB9F0((__int64)a1, a3, 0, 0);
      v31 = *(_DWORD *)(v8 + 8);
      if ( v31 > 0x40 )
        sub_C43780((__int64)&v30, (const void **)v8);
      else
        v30 = *(const void **)v8;
      v33 = *(_DWORD *)(v8 + 24);
      if ( v33 > 0x40 )
        sub_C43780((__int64)&v32, (const void **)(v8 + 16));
      else
        v32 = *(_QWORD *)(v8 + 16);
      v9 = 0;
    }
    v10 = sub_DBB9F0((__int64)a1, a4, v9, 0);
    v35 = *(_DWORD *)(v10 + 8);
    if ( v35 > 0x40 )
    {
      sub_C43780((__int64)&v34, (const void **)v10);
      v37 = *(_DWORD *)(v10 + 24);
      if ( v37 <= 0x40 )
        goto LABEL_12;
    }
    else
    {
      v34 = *(const void **)v10;
      v37 = *(_DWORD *)(v10 + 24);
      if ( v37 <= 0x40 )
      {
LABEL_12:
        v36 = *(_QWORD *)(v10 + 16);
LABEL_13:
        LOBYTE(v11) = sub_ABB410((__int64 *)&v30, a2, (__int64 *)&v34);
        v7 = v11;
        if ( v37 > 0x40 && v36 )
          j_j___libc_free_0_0(v36);
        if ( v35 > 0x40 && v34 )
          j_j___libc_free_0_0(v34);
        if ( v33 > 0x40 && v32 )
          j_j___libc_free_0_0(v32);
        if ( v31 <= 0x40 )
          return v7;
        v12 = v30;
        if ( !v30 )
          return v7;
        goto LABEL_24;
      }
    }
    sub_C43780((__int64)&v36, (const void **)(v10 + 16));
    goto LABEL_13;
  }
  return sub_B535D0(a2);
}
