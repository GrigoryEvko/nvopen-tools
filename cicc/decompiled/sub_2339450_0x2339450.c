// Function: sub_2339450
// Address: 0x2339450
//
__int64 __fastcall sub_2339450(char *a1, _QWORD *a2, __int64 *a3)
{
  _QWORD *v5; // r12
  char *v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  size_t v9; // r15
  __int64 v10; // rdx
  unsigned int v11; // r15d
  unsigned int v12; // eax
  __int64 v13; // rdi
  _QWORD *v14; // rcx
  unsigned int v15; // eax
  unsigned int v16; // r14d
  _QWORD *v17; // r12
  _QWORD *i; // rbx
  _QWORD *v20; // r12
  _QWORD *j; // rbx
  _QWORD *v22; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+18h] [rbp-88h]
  _QWORD v24[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v25[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v26; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v27; // [rsp+48h] [rbp-58h]
  __int64 v28; // [rsp+50h] [rbp-50h]
  __int64 v29; // [rsp+58h] [rbp-48h]
  __int64 v30; // [rsp+60h] [rbp-40h]

  v5 = a2;
  if ( a2 )
  {
    v6 = a1;
    while ( !sub_2304E30((__int64)&v26, *v6) )
    {
      v6 = (char *)(v8 + 1);
      if ( v7 == 1 )
      {
        v9 = (size_t)a2;
        goto LABEL_7;
      }
    }
    v9 = (size_t)a2 - v7;
    if ( (_QWORD *)((char *)a2 - v7) > a2 )
      v9 = (size_t)a2;
  }
  else
  {
    v9 = 0;
  }
LABEL_7:
  if ( sub_9691B0(a1, (size_t)a2, "cgscc", 5) )
    return 1;
  if ( sub_9691B0(a1, v9, "function", 8) )
    return 1;
  v26 = (_QWORD *)sub_232E0D0((__int64)a1, (__int64)a2);
  if ( BYTE4(v26) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<no-op-cgscc>", 20) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<no-op-cgscc>", 23) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<fam-proxy>", 18) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<fam-proxy>", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<pass-instrumentation>", 29) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<pass-instrumentation>", 32) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "argpromotion", 12) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "attributor-cgscc", 16) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "attributor-light-cgscc", 22) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<all>", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "no-op-cgscc", 11) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "openmp-opt-cgscc", 16) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "coro-annotation-elide", 21) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "coro-split", 0xAu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "function-attrs", 0xEu) )
    return 1;
  v11 = sub_2337DE0(a1, (__int64)a2, "inline", 6u);
  if ( (_BYTE)v11 )
  {
    return 1;
  }
  else
  {
    v12 = *((_DWORD *)a3 + 2);
    if ( v12 )
    {
      v13 = *a3;
      v26 = 0;
      v14 = v25;
      v27 = 0;
      v28 = 0;
      v29 = 0;
      v30 = 0;
      v23 = v13 + 32LL * v12;
      do
      {
        if ( v23 == v13 )
        {
          v17 = v27;
          for ( i = v26; v17 != i; ++i )
          {
            if ( *i )
              (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*i + 8LL))(*i);
          }
          if ( v26 )
            j_j___libc_free_0((unsigned __int64)v26);
          return v11;
        }
        v24[0] = a1;
        v24[1] = v5;
        v25[0] = 0;
        v25[1] = 0;
        if ( !*(_QWORD *)(v13 + 16) )
          sub_4263D6(v13, a2, v10);
        v22 = v14;
        a2 = v24;
        v15 = (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD **))(v13 + 24))(v13, v24, &v26);
        v14 = v22;
        v16 = v15;
        v13 += 32;
      }
      while ( !(_BYTE)v15 );
      v20 = v27;
      for ( j = v26; v20 != j; ++j )
      {
        if ( *j )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*j + 8LL))(*j);
      }
      if ( v26 )
        j_j___libc_free_0((unsigned __int64)v26);
      return v16;
    }
  }
  return v11;
}
