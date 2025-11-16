// Function: sub_AC10C0
// Address: 0xac10c0
//
_QWORD *__fastcall sub_AC10C0(_QWORD *a1, unsigned int *a2, __int64 *a3)
{
  __int64 *v6; // r13
  __int64 *v7; // rbx
  __int64 *v8; // rsi
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r8
  int v14; // eax
  __int64 *v15; // rsi
  __int64 v16; // r12
  unsigned __int64 v17; // rdx
  char *v18; // rbx
  unsigned __int64 *v19; // rsi
  int v20; // eax
  unsigned __int64 v21; // r12
  unsigned int v22; // eax
  unsigned int v23; // eax
  unsigned __int64 v24; // rbx
  _BYTE *v25; // r12
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rdx
  char *v29; // rbx
  __int64 v30; // rdx
  __int64 v31; // rdx
  _QWORD *v32; // [rsp+20h] [rbp-D0h]
  __int64 v34; // [rsp+30h] [rbp-C0h]
  __int64 v35; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v36[2]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v37; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v38; // [rsp+58h] [rbp-98h]
  __int64 v39; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v40; // [rsp+68h] [rbp-88h]
  unsigned __int64 v41; // [rsp+70h] [rbp-80h] BYREF
  __int64 v42; // [rsp+78h] [rbp-78h]
  _BYTE v43[112]; // [rsp+80h] [rbp-70h] BYREF

  v32 = a1 + 2;
  if ( a2[2] )
  {
    if ( *((_DWORD *)a3 + 2) )
    {
      v41 = (unsigned __int64)v43;
      v42 = 0x200000000LL;
      sub_AADB10((__int64)&v37, *(_DWORD *)(*(_QWORD *)a2 + 8LL), 0);
      v6 = *(__int64 **)a2;
      v7 = (__int64 *)*a3;
      if ( (int)sub_C4C880(*(_QWORD *)a2, *a3) < 0 )
      {
        if ( v38 <= 0x40 && *((_DWORD *)v6 + 2) <= 0x40u )
        {
          v30 = *v6;
          v38 = *((_DWORD *)v6 + 2);
          v37 = v30;
        }
        else
        {
          sub_C43990(&v37, v6);
        }
        v15 = v6 + 2;
        if ( v40 <= 0x40 && *((_DWORD *)v6 + 6) <= 0x40u )
        {
          v31 = v6[2];
          v9 = 0;
          v40 = *((_DWORD *)v6 + 6);
          v10 = 1;
          v39 = v31;
        }
        else
        {
          v9 = 0;
          v10 = 1;
          sub_C43990(&v39, v15);
        }
      }
      else
      {
        if ( v38 <= 0x40 && *((_DWORD *)v7 + 2) <= 0x40u )
        {
          v27 = *v7;
          v38 = *((_DWORD *)v7 + 2);
          v37 = v27;
        }
        else
        {
          sub_C43990(&v37, v7);
        }
        v8 = v7 + 2;
        if ( v40 <= 0x40 && *((_DWORD *)v7 + 6) <= 0x40u )
        {
          v28 = v7[2];
          v10 = 0;
          v40 = *((_DWORD *)v7 + 6);
          v9 = 1;
          v39 = v28;
        }
        else
        {
          v9 = 1;
          v10 = 0;
          sub_C43990(&v39, v8);
        }
      }
      v36[0] = (unsigned __int64)&v37;
      v36[1] = (unsigned __int64)&v41;
      while ( 1 )
      {
        v11 = *((unsigned int *)a3 + 2);
        if ( v10 >= a2[2] )
          break;
        while ( 1 )
        {
          v13 = *(_QWORD *)a2 + 32 * v10;
          if ( v9 == v11 )
            break;
          v35 = *(_QWORD *)a2 + 32 * v10;
          v34 = *a3 + 32 * v9;
          v14 = sub_C4C880(v35, v34);
          v13 = v35;
          v12 = v34;
          if ( v14 < 0 )
            break;
LABEL_17:
          ++v9;
          sub_AC0DB0(v36, v12);
          v11 = *((unsigned int *)a3 + 2);
          if ( v10 >= a2[2] )
            goto LABEL_15;
        }
        ++v10;
        sub_AC0DB0(v36, v13);
      }
LABEL_15:
      if ( v9 < v11 )
      {
        v12 = *a3 + 32 * v9;
        goto LABEL_17;
      }
      v16 = (unsigned int)v42;
      v17 = v41;
      v18 = (char *)&v37;
      v19 = (unsigned __int64 *)((unsigned int)v42 + 1LL);
      v20 = v42;
      if ( (unsigned __int64)v19 > HIDWORD(v42) )
      {
        if ( v41 > (unsigned __int64)&v37 || (unsigned __int64)&v37 >= v41 + 32LL * (unsigned int)v42 )
        {
          sub_9D5330((__int64)&v41, (__int64)v19);
          v16 = (unsigned int)v42;
          v17 = v41;
          v20 = v42;
        }
        else
        {
          v29 = (char *)&v37 - v41;
          sub_9D5330((__int64)&v41, (__int64)v19);
          v17 = v41;
          v16 = (unsigned int)v42;
          v18 = &v29[v41];
          v20 = v42;
        }
      }
      v21 = v17 + 32 * v16;
      if ( v21 )
      {
        v22 = *((_DWORD *)v18 + 2);
        *(_DWORD *)(v21 + 8) = v22;
        if ( v22 > 0x40 )
        {
          v19 = (unsigned __int64 *)v18;
          sub_C43780(v21, v18);
        }
        else
        {
          *(_QWORD *)v21 = *(_QWORD *)v18;
        }
        v23 = *((_DWORD *)v18 + 6);
        *(_DWORD *)(v21 + 24) = v23;
        if ( v23 > 0x40 )
        {
          v19 = (unsigned __int64 *)(v18 + 16);
          sub_C43780(v21 + 16, v18 + 16);
        }
        else
        {
          *(_QWORD *)(v21 + 16) = *((_QWORD *)v18 + 2);
        }
        v20 = v42;
      }
      LODWORD(v42) = v20 + 1;
      *a1 = v32;
      a1[1] = 0x200000000LL;
      if ( v20 != -1 )
      {
        v19 = &v41;
        sub_ABF400((__int64)a1, &v41);
      }
      if ( v40 > 0x40 && v39 )
        j_j___libc_free_0_0(v39);
      if ( v38 > 0x40 && v37 )
        j_j___libc_free_0_0(v37);
      v24 = v41;
      v25 = (_BYTE *)(v41 + 32LL * (unsigned int)v42);
      if ( (_BYTE *)v41 != v25 )
      {
        do
        {
          v25 -= 32;
          if ( *((_DWORD *)v25 + 6) > 0x40u )
          {
            v26 = *((_QWORD *)v25 + 2);
            if ( v26 )
              j_j___libc_free_0_0(v26);
          }
          if ( *((_DWORD *)v25 + 2) > 0x40u && *(_QWORD *)v25 )
            j_j___libc_free_0_0(*(_QWORD *)v25);
        }
        while ( (_BYTE *)v24 != v25 );
        v25 = (_BYTE *)v41;
      }
      if ( v25 != v43 )
        _libc_free(v25, v19);
    }
    else
    {
      *a1 = v32;
      a1[1] = 0x200000000LL;
      if ( a2[2] )
        sub_ABF850((__int64)a1, (__int64 *)a2);
    }
  }
  else
  {
    *a1 = v32;
    a1[1] = 0x200000000LL;
    if ( *((_DWORD *)a3 + 2) )
      sub_ABF850((__int64)a1, a3);
  }
  return a1;
}
