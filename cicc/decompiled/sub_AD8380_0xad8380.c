// Function: sub_AD8380
// Address: 0xad8380
//
__int64 __fastcall sub_AD8380(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v5; // rbx
  __int64 v6; // rdx
  unsigned int v7; // r15d
  _BYTE *v8; // rax
  unsigned int v9; // ebx
  unsigned int v10; // eax
  unsigned int v11; // eax
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned int v14; // eax
  int v15; // [rsp+8h] [rbp-B8h]
  __int64 v16; // [rsp+8h] [rbp-B8h]
  __int64 v17; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-98h]
  __int64 v19; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-88h]
  __int64 v21; // [rsp+40h] [rbp-80h]
  unsigned int v22; // [rsp+48h] [rbp-78h]
  __int64 v23; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+58h] [rbp-68h]
  __int64 v25; // [rsp+60h] [rbp-60h]
  unsigned int v26; // [rsp+68h] [rbp-58h]
  __int64 v27; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+78h] [rbp-48h]
  __int64 v29; // [rsp+80h] [rbp-40h]
  unsigned int v30; // [rsp+88h] [rbp-38h]

  if ( *(_BYTE *)a2 == 17 )
  {
    v28 = *(_DWORD *)(a2 + 32);
    if ( v28 > 0x40 )
      sub_C43780(&v27, a2 + 24);
    else
      v27 = *(_QWORD *)(a2 + 24);
    goto LABEL_4;
  }
  v5 = *(_QWORD *)(a2 + 8);
  v7 = sub_BCB060(v5);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 > 1 )
    goto LABEL_14;
  v8 = sub_AD7630(a2, 1, v6);
  if ( !v8 || *v8 != 17 )
  {
    if ( *(_BYTE *)a2 == 16 )
    {
      sub_AADB10((__int64)&v19, v7, 0);
      v15 = sub_AC5290(a2);
      if ( v15 )
      {
        v9 = 0;
        sub_AC5390((__int64)&v17, a2, 0);
        while ( 1 )
        {
          sub_AADBC0((__int64)&v23, &v17);
          sub_AB3510((__int64)&v27, (__int64)&v19, (__int64)&v23, 0);
          if ( v20 > 0x40 && v19 )
            j_j___libc_free_0_0(v19);
          v19 = v27;
          v10 = v28;
          v28 = 0;
          v20 = v10;
          if ( v22 > 0x40 && v21 )
          {
            j_j___libc_free_0_0(v21);
            v21 = v29;
            v22 = v30;
            if ( v28 > 0x40 && v27 )
              j_j___libc_free_0_0(v27);
          }
          else
          {
            v21 = v29;
            v22 = v30;
          }
          if ( v26 > 0x40 && v25 )
            j_j___libc_free_0_0(v25);
          if ( v24 > 0x40 && v23 )
            j_j___libc_free_0_0(v23);
          if ( v18 > 0x40 && v17 )
            j_j___libc_free_0_0(v17);
          if ( v15 == ++v9 )
            break;
          sub_AC5390((__int64)&v17, a2, v9);
        }
      }
      goto LABEL_39;
    }
    if ( *(_BYTE *)a2 == 11 )
    {
      sub_AADB10((__int64)&v19, v7, 0);
      v11 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      if ( v11 )
      {
        v12 = 0;
        v16 = v11 - 1;
        while ( 1 )
        {
          v13 = *(_QWORD *)(a2 + 32 * (v12 - v11));
          if ( !v13 )
            break;
          if ( *(_BYTE *)v13 != 13 )
          {
            if ( *(_BYTE *)v13 != 17 )
              break;
            v18 = *(_DWORD *)(v13 + 32);
            if ( v18 > 0x40 )
              sub_C43780(&v17, v13 + 24);
            else
              v17 = *(_QWORD *)(v13 + 24);
            sub_AADBC0((__int64)&v23, &v17);
            sub_AB3510((__int64)&v27, (__int64)&v19, (__int64)&v23, 0);
            if ( v20 > 0x40 && v19 )
              j_j___libc_free_0_0(v19);
            v19 = v27;
            v14 = v28;
            v28 = 0;
            v20 = v14;
            if ( v22 > 0x40 && v21 )
            {
              j_j___libc_free_0_0(v21);
              v21 = v29;
              v22 = v30;
              if ( v28 > 0x40 && v27 )
                j_j___libc_free_0_0(v27);
            }
            else
            {
              v21 = v29;
              v22 = v30;
            }
            if ( v26 > 0x40 && v25 )
              j_j___libc_free_0_0(v25);
            if ( v24 > 0x40 && v23 )
              j_j___libc_free_0_0(v23);
            if ( v18 > 0x40 && v17 )
              j_j___libc_free_0_0(v17);
          }
          if ( v16 == v12 )
            goto LABEL_39;
          ++v12;
          v11 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
        }
        sub_AADB10(a1, v7, 1);
        if ( v22 > 0x40 && v21 )
          j_j___libc_free_0_0(v21);
        if ( v20 > 0x40 )
        {
          v3 = v19;
          if ( v19 )
            goto LABEL_6;
        }
        return a1;
      }
LABEL_39:
      *(_DWORD *)(a1 + 8) = v20;
      *(_QWORD *)a1 = v19;
      *(_DWORD *)(a1 + 24) = v22;
      *(_QWORD *)(a1 + 16) = v21;
      return a1;
    }
LABEL_14:
    sub_AADB10(a1, v7, 1);
    return a1;
  }
  v28 = *((_DWORD *)v8 + 8);
  if ( v28 > 0x40 )
    sub_C43780(&v27, v8 + 24);
  else
    v27 = *((_QWORD *)v8 + 3);
LABEL_4:
  sub_AADBC0(a1, &v27);
  if ( v28 > 0x40 )
  {
    v3 = v27;
    if ( v27 )
LABEL_6:
      j_j___libc_free_0_0(v3);
  }
  return a1;
}
