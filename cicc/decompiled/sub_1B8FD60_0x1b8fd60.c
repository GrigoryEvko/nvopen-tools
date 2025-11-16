// Function: sub_1B8FD60
// Address: 0x1b8fd60
//
__int64 __fastcall sub_1B8FD60(__int64 a1, unsigned int a2, __int64 *a3, __int64 *a4, _BYTE *a5)
{
  unsigned int v5; // r12d
  __int64 v6; // rbx
  __int64 v7; // r13
  const char *v8; // rax
  int v9; // r9d
  bool v10; // sf
  __int64 *v11; // r15
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r14
  int v16; // r14d
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r10
  __int64 ***v21; // r10
  __int64 ***v22; // r14
  __int64 ***v23; // r12
  __int64 *v24; // rbx
  unsigned int v25; // r13d
  int v26; // r9d
  __int64 **v27; // r14
  __int64 **v28; // r15
  __int64 *v29; // r8
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v34; // rax
  unsigned int v35; // eax
  __int64 v36; // [rsp+0h] [rbp-F0h]
  char *v37; // [rsp+8h] [rbp-E8h]
  __int64 *v39; // [rsp+20h] [rbp-D0h]
  __int64 v42; // [rsp+58h] [rbp-98h] BYREF
  _BYTE *v43; // [rsp+60h] [rbp-90h] BYREF
  __int64 v44; // [rsp+68h] [rbp-88h]
  _BYTE v45[32]; // [rsp+70h] [rbp-80h] BYREF
  __int64 **v46; // [rsp+90h] [rbp-60h] BYREF
  __int64 v47; // [rsp+98h] [rbp-58h]
  _BYTE v48[80]; // [rsp+A0h] [rbp-50h] BYREF

  v5 = a2;
  v6 = a1;
  v7 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v7 + 16) )
    v7 = 0;
  v8 = sub_1649960(v7);
  v10 = *(char *)(a1 + 23) < 0;
  v11 = *(__int64 **)a1;
  v37 = (char *)v8;
  v43 = v45;
  v36 = v12;
  v44 = 0x400000000LL;
  v46 = (__int64 **)v48;
  v47 = 0x400000000LL;
  if ( !v10 )
  {
    v18 = 0;
    v20 = -24;
    goto LABEL_9;
  }
  v13 = sub_1648A40(a1);
  v15 = v13 + v14;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( (unsigned int)(v15 >> 4) )
LABEL_40:
      BUG();
LABEL_34:
    v18 = (unsigned int)v47;
    v20 = -24;
    goto LABEL_9;
  }
  if ( !(unsigned int)((v15 - sub_1648A40(a1)) >> 4) )
    goto LABEL_34;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_40;
  v16 = *(_DWORD *)(sub_1648A40(a1) + 8);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  v17 = sub_1648A40(a1);
  v18 = (unsigned int)v47;
  v20 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v17 + v19 - 4) - v16);
LABEL_9:
  v21 = (__int64 ***)(a1 + v20);
  v22 = (__int64 ***)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( v21 != v22 )
  {
    v23 = v21;
    do
    {
      v24 = **v22;
      if ( HIDWORD(v47) <= (unsigned int)v18 )
      {
        sub_16CD150((__int64)&v46, v48, 0, 8, v18, v9);
        v18 = (unsigned int)v47;
      }
      v22 += 3;
      v46[v18] = v24;
      v18 = (unsigned int)(v47 + 1);
      LODWORD(v47) = v47 + 1;
    }
    while ( v23 != v22 );
    v6 = a1;
    v5 = a2;
  }
  v25 = sub_14A35C0((__int64)a3);
  if ( v5 != 1 )
  {
    sub_1B8E090(v11, v5);
    v27 = v46;
    v28 = &v46[(unsigned int)v47];
    if ( v46 != v28 )
    {
      do
      {
        v29 = *v27;
        if ( *((_BYTE *)*v27 + 8) )
          v29 = sub_16463B0(*v27, v5);
        v30 = (unsigned int)v44;
        if ( (unsigned int)v44 >= HIDWORD(v44) )
        {
          v39 = v29;
          sub_16CD150((__int64)&v43, v45, 0, 8, (int)v29, v26);
          v30 = (unsigned int)v44;
          v29 = v39;
        }
        ++v27;
        *(_QWORD *)&v43[8 * v30] = v29;
        LODWORD(v44) = v44 + 1;
      }
      while ( v28 != v27 );
    }
    v25 = sub_1B8FA60(v6, v5, a3) + v5 * v25;
    *a5 = 1;
    if ( a4 )
    {
      sub_149D1A0(*a4, v37, v36, v5);
      if ( v31 )
      {
        if ( !(unsigned __int8)sub_1560260((_QWORD *)(v6 + 56), -1, 21)
          && ((v34 = *(_QWORD *)(v6 - 24), *(_BYTE *)(v34 + 16))
           || (v42 = *(_QWORD *)(v34 + 112), !(unsigned __int8)sub_1560260(&v42, -1, 21)))
          || (unsigned __int8)sub_1560260((_QWORD *)(v6 + 56), -1, 5)
          || (v32 = *(_QWORD *)(v6 - 24), !*(_BYTE *)(v32 + 16))
          && (v42 = *(_QWORD *)(v32 + 112), (unsigned __int8)sub_1560260(&v42, -1, 5)) )
        {
          v35 = sub_14A35C0((__int64)a3);
          if ( v25 > v35 )
          {
            v25 = v35;
            *a5 = 0;
          }
        }
      }
    }
  }
  if ( v46 != (__int64 **)v48 )
    _libc_free((unsigned __int64)v46);
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
  return v25;
}
