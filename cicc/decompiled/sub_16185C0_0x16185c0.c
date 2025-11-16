// Function: sub_16185C0
// Address: 0x16185c0
//
_QWORD *__fastcall sub_16185C0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  __int64 v5; // rax
  __int64 v6; // rbx
  _QWORD *v7; // r15
  __int64 v8; // r13
  _QWORD *v9; // rax
  _QWORD *result; // rax
  void (__fastcall *v11)(__int64 *, __int64, _QWORD); // r15
  unsigned int v12; // eax
  __int64 *v13; // rdi
  __int64 v14; // rdx
  _BYTE *v15; // rsi
  __int64 v16; // rax
  _QWORD *v17; // r15
  __int64 (__fastcall *v18)(_QWORD *, __int64, _QWORD); // r14
  unsigned int v19; // eax
  __int64 v20; // r14
  __int64 (__fastcall *v21)(__int64, __int64, _QWORD); // rbx
  unsigned int v22; // eax
  __int64 (__fastcall *v23)(__int64 *, __int64, __int64 *); // [rsp+10h] [rbp-70h]
  __int64 v24; // [rsp+28h] [rbp-58h] BYREF
  __int64 v25[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v26[8]; // [rsp+40h] [rbp-40h] BYREF

  v3 = a1 + 8;
  (*(void (__fastcall **)(__int64 *, __int64))(*a2 + 72))(a2, a1 + 8);
  v5 = sub_1614F20(a1, a2[2]);
  v6 = v5;
  if ( v5 && *(_BYTE *)(v5 + 41) && sub_160EA80(a1, a2[2]) )
    return (_QWORD *)(*(__int64 (__fastcall **)(__int64 *))(*a2 + 8))(a2);
  v24 = a1;
  sub_1618840(&v24, a2);
  v7 = (_QWORD *)(*(__int64 (__fastcall **)(__int64 *))(*a2 + 112))(a2);
  if ( v7 )
  {
    v8 = (**(__int64 (__fastcall ***)(__int64))a1)(a1);
    v9 = (_QWORD *)sub_22077B0(32);
    if ( v9 )
    {
      *v9 = 0;
      v9[1] = 0;
      v9[2] = 0;
      v9[3] = v8;
    }
    sub_1636870(a2, v9);
    sub_1614C80(v8, (__int64)a2);
    sub_16164C0(a1, v7);
    return sub_16176C0(v8, (__int64)v7);
  }
  else
  {
    v11 = *(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a2 + 64);
    v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 8LL))(a1);
    v11(a2, v3, v12);
    result = qword_4F9E580;
    if ( qword_4F9E580[20] == qword_4F9E580[21] )
    {
      if ( !v6 )
        return result;
    }
    else
    {
      if ( !v6 )
        return result;
      if ( !*(_BYTE *)(v6 + 41) )
      {
        v13 = a2;
        v23 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64 *))(*a2 + 56);
        v15 = (_BYTE *)(*(__int64 (__fastcall **)(__int64 *))(*a2 + 16))(a2);
        if ( v15 )
        {
          v13 = v25;
          v25[0] = (__int64)v26;
          sub_160D9C0(v25, v15, (__int64)&v15[v14]);
        }
        else
        {
          v25[1] = 0;
          v25[0] = (__int64)v26;
          LOBYTE(v26[0]) = 0;
        }
        v16 = sub_16BA580(v13, v15, v14);
        result = (_QWORD *)v23(a2, v16, v25);
        v17 = result;
        if ( (_QWORD *)v25[0] != v26 )
          result = (_QWORD *)j_j___libc_free_0(v25[0], v26[0] + 1LL);
        if ( v17 )
        {
          sub_1618840(&v24, v17);
          v18 = *(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*v17 + 64LL);
          v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 8LL))(a1);
          result = (_QWORD *)v18(v17, v3, v19);
        }
      }
    }
    if ( byte_4F9E8A0 && !*(_BYTE *)(v6 + 41) )
    {
      v20 = sub_1654860(1);
      v21 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v20 + 64LL);
      v22 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 8LL))(a1);
      return (_QWORD *)v21(v20, v3, v22);
    }
  }
  return result;
}
