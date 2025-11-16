// Function: sub_9E1AB0
// Address: 0x9e1ab0
//
_QWORD *__fastcall sub_9E1AB0(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // rbx
  _OWORD *v5; // r9
  _OWORD *v7; // r12
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  bool v12; // zf
  unsigned __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rcx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 i; // r12
  __int64 v23; // rdi
  unsigned int v24; // eax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v32; // [rsp+0h] [rbp-A0h]
  __int64 v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+8h] [rbp-98h]
  _QWORD *v37; // [rsp+28h] [rbp-78h]
  __int64 v38; // [rsp+28h] [rbp-78h]
  unsigned __int64 v39; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v40; // [rsp+38h] [rbp-68h]
  unsigned __int64 v41; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v42; // [rsp+48h] [rbp-58h]
  __int64 v43; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v44; // [rsp+58h] [rbp-48h]
  __int64 v45; // [rsp+60h] [rbp-40h]
  int v46; // [rsp+68h] [rbp-38h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( a4 )
  {
    v4 = a4;
    v5 = 0;
LABEL_65:
    sub_9CCC50((__int64)a1, (__int64)v5);
    v7 = (_OWORD *)a1[1];
    while ( 1 )
    {
      *((_QWORD *)v7 - 8) = *a3;
      v8 = a3[1];
      v9 = v8 >> 1;
      if ( (v8 & 1) != 0 )
      {
        v9 = -(__int64)v9;
        if ( v8 == 1 )
          v9 = 0x8000000000000000LL;
      }
      v10 = a3[2];
      if ( (v10 & 1) != 0 )
      {
        v11 = -(__int64)(v10 >> 1);
        v12 = v10 == 1;
        v13 = 0x8000000000000000LL;
        if ( !v12 )
          v13 = v11;
      }
      else
      {
        v13 = v10 >> 1;
      }
      v39 = v13;
      v42 = 64;
      v41 = v9;
      v40 = 64;
      sub_AADC30(&v43, &v41, &v39);
      if ( v40 > 0x40 && v39 )
        j_j___libc_free_0_0(v39);
      if ( v42 > 0x40 && v41 )
        j_j___libc_free_0_0(v41);
      if ( *((_DWORD *)v7 - 12) > 0x40u )
      {
        v14 = *((_QWORD *)v7 - 7);
        if ( v14 )
          j_j___libc_free_0_0(v14);
      }
      *((_QWORD *)v7 - 7) = v43;
      *((_DWORD *)v7 - 12) = v44;
      v44 = 0;
      if ( *((_DWORD *)v7 - 8) > 0x40u )
      {
        v15 = *((_QWORD *)v7 - 5);
        if ( v15 )
          j_j___libc_free_0_0(v15);
      }
      *((_QWORD *)v7 - 5) = v45;
      *((_DWORD *)v7 - 8) = v46;
      if ( v44 > 0x40 && v43 )
        j_j___libc_free_0_0(v43);
      v16 = *((_QWORD *)v7 - 3);
      v17 = a3[3];
      v38 = *((_QWORD *)v7 - 2);
      v18 = 0xAAAAAAAAAAAAAAABLL * ((v38 - v16) >> 4);
      if ( v17 > v18 )
      {
        sub_9CD190((__int64 *)v7 - 3, v17 - v18);
        v16 = *((_QWORD *)v7 - 3);
        v38 = *((_QWORD *)v7 - 2);
      }
      else if ( v17 < v18 )
      {
        v32 = v16 + 48 * v17;
        if ( v38 != v32 )
        {
          v19 = v16 + 48 * v17;
          do
          {
            if ( *(_DWORD *)(v19 + 40) > 0x40u )
            {
              v20 = *(_QWORD *)(v19 + 32);
              if ( v20 )
              {
                v33 = v19;
                j_j___libc_free_0_0(v20);
                v19 = v33;
              }
            }
            if ( *(_DWORD *)(v19 + 24) > 0x40u )
            {
              v21 = *(_QWORD *)(v19 + 16);
              if ( v21 )
              {
                v34 = v19;
                j_j___libc_free_0_0(v21);
                v19 = v34;
              }
            }
            v19 += 48;
          }
          while ( v38 != v19 );
          v16 = *((_QWORD *)v7 - 3);
          *((_QWORD *)v7 - 2) = v32;
          v38 = v32;
        }
      }
      v4 -= 4;
      a3 += 4;
      for ( i = v16; v38 != i; i += 48 )
      {
        while ( 1 )
        {
          *(_QWORD *)i = *a3;
          *(_QWORD *)(i + 8) = sub_9E1590(a2, *((_DWORD *)a3 + 2));
          v25 = a3[2];
          if ( (v25 & 1) != 0 )
          {
            v26 = -(__int64)(v25 >> 1);
            if ( v25 == 1 )
              v26 = 0x8000000000000000LL;
          }
          else
          {
            v26 = v25 >> 1;
          }
          v27 = a3[3];
          if ( (v27 & 1) != 0 )
          {
            v28 = -(__int64)(v27 >> 1);
            v12 = v27 == 1;
            v29 = 0x8000000000000000LL;
            if ( !v12 )
              v29 = v28;
          }
          else
          {
            v29 = v27 >> 1;
          }
          v41 = v26;
          v42 = 64;
          v4 -= 4;
          a3 += 4;
          v40 = 64;
          v39 = v29;
          sub_AADC30(&v43, &v41, &v39);
          if ( v40 > 0x40 && v39 )
            j_j___libc_free_0_0(v39);
          if ( v42 > 0x40 && v41 )
            j_j___libc_free_0_0(v41);
          if ( *(_DWORD *)(i + 24) > 0x40u )
          {
            v30 = *(_QWORD *)(i + 16);
            if ( v30 )
              j_j___libc_free_0_0(v30);
          }
          *(_QWORD *)(i + 16) = v43;
          *(_DWORD *)(i + 24) = v44;
          v44 = 0;
          if ( *(_DWORD *)(i + 40) > 0x40u )
          {
            v23 = *(_QWORD *)(i + 32);
            if ( v23 )
              break;
          }
          i += 48;
          *(_QWORD *)(i - 16) = v45;
          *(_DWORD *)(i - 8) = v46;
          if ( v38 == i )
            goto LABEL_63;
        }
        j_j___libc_free_0_0(v23);
        v24 = v44;
        *(_QWORD *)(i + 32) = v45;
        *(_DWORD *)(i + 40) = v46;
        if ( v24 > 0x40 && v43 )
          j_j___libc_free_0_0(v43);
      }
LABEL_63:
      if ( !v4 )
        break;
      v5 = (_OWORD *)a1[1];
      if ( v5 == (_OWORD *)a1[2] )
        goto LABEL_65;
      if ( v5 )
      {
        *v5 = 0;
        v5[1] = 0;
        v5[2] = 0;
        v5[3] = 0;
        v37 = v5;
        sub_AADB10((char *)v5 + 8, 64, 1);
        v37[5] = 0;
        v37[6] = 0;
        v37[7] = 0;
        v5 = (_OWORD *)a1[1];
      }
      v7 = v5 + 4;
      a1[1] = v5 + 4;
    }
  }
  return a1;
}
