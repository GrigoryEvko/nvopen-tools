// Function: sub_AB4490
// Address: 0xab4490
//
__int64 __fastcall sub_AB4490(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v5; // ebx
  unsigned int v6; // eax
  int v7; // eax
  __int64 v8; // r8
  int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // rax
  unsigned int v17; // edx
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // [rsp+8h] [rbp-C8h]
  __int64 v21; // [rsp+10h] [rbp-C0h]
  unsigned int v22; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v23; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-A8h]
  void *s; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-98h]
  __int64 v27[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v28; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v29; // [rsp+58h] [rbp-78h]
  __int64 v30; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v31; // [rsp+68h] [rbp-68h]
  __int64 v32; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v33; // [rsp+78h] [rbp-58h]
  __int64 v34; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v35; // [rsp+88h] [rbp-48h]
  __int64 v36; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v37; // [rsp+98h] [rbp-38h]

  if ( !sub_AAF7D0(a2) )
  {
    if ( sub_AAF760(a2) )
    {
      sub_AADB10(a1, a3, 1);
      return a1;
    }
    sub_9865C0((__int64)&v23, a2);
    v21 = a2 + 16;
    sub_9865C0((__int64)&s, a2 + 16);
    sub_AADB10((__int64)&v30, a3, 0);
    if ( !sub_AB0100(a2) )
    {
      v22 = v24;
      goto LABEL_8;
    }
    v20 = *(_DWORD *)(a2 + 24);
    v7 = sub_9871A0(v21);
    v8 = a2 + 16;
    if ( a3 < v20 - v7 )
      goto LABEL_11;
    if ( v20 <= 0x40 )
    {
      v9 = 64;
      _RDX = ~*(_QWORD *)(a2 + 16);
      __asm { tzcnt   rcx, rdx }
      if ( *(_QWORD *)(a2 + 16) != -1 )
        v9 = _RCX;
    }
    else
    {
      v9 = sub_C445E0(v21);
      v8 = a2 + 16;
    }
    if ( a3 == v9 )
    {
LABEL_11:
      sub_AADB10(a1, a3, 1);
LABEL_12:
      sub_969240(&v32);
      sub_969240(&v30);
      sub_969240((__int64 *)&s);
      sub_969240((__int64 *)&v23);
      return a1;
    }
    sub_C44740(&v28, v8);
    sub_9691E0((__int64)v27, a3, -1, 1u, 0);
    sub_AADC30((__int64)&v34, (__int64)v27, &v28);
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
    v30 = v34;
    v10 = v35;
    v35 = 0;
    v31 = v10;
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    v32 = v36;
    v11 = v37;
    v37 = 0;
    v33 = v11;
    sub_969240(&v36);
    sub_969240(&v34);
    sub_969240(v27);
    sub_969240(&v28);
    if ( v26 > 0x40 )
      memset(s, -1, 8 * (((unsigned __int64)v26 + 63) >> 6));
    else
      s = (void *)-1LL;
    sub_AAD510((unsigned __int64 *)&s);
    v22 = v24;
    if ( v24 <= 0x40 )
    {
      if ( (void *)v23 == s )
        goto LABEL_27;
    }
    else if ( (unsigned __int8)sub_C43C50(&v23, &s) )
    {
LABEL_27:
      v12 = v31;
      v31 = 0;
      *(_DWORD *)(a1 + 8) = v12;
      *(_QWORD *)a1 = v30;
      v13 = v33;
      v33 = 0;
      *(_DWORD *)(a1 + 24) = v13;
      *(_QWORD *)(a1 + 16) = v32;
      goto LABEL_12;
    }
LABEL_8:
    if ( a3 >= v22 - (unsigned int)sub_9871A0((__int64)&v23) )
    {
LABEL_9:
      v5 = v26;
      v6 = v5 - sub_9871A0((__int64)&s);
      if ( a3 >= v6
        || a3 + 1 == v6
        && ((v14 = ~(1LL << a3), v5 > 0x40)
          ? (void *)(*((_QWORD *)s + (a3 >> 6)) &= v14)
          : (s = (void *)((unsigned __int64)s & v14)),
            (int)sub_C49970(&s, &v23) < 0) )
      {
        sub_C44740(&v28, &s);
        sub_C44740(v27, &v23);
        sub_AADC30((__int64)&v34, (__int64)v27, &v28);
        sub_AB3510(a1, (__int64)&v34, (__int64)&v30, 0);
        sub_969240(&v36);
        sub_969240(&v34);
        sub_969240(v27);
        sub_969240(&v28);
        goto LABEL_12;
      }
      goto LABEL_11;
    }
    sub_9691E0((__int64)&v34, *(_DWORD *)(a2 + 8), 0, 0, 0);
    v17 = v35;
    if ( a3 != v35 )
    {
      if ( a3 <= 0x3F && v35 <= 0x40 )
      {
        v18 = v34 | (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)a3 - (unsigned __int8)v35 + 64) << a3);
        goto LABEL_39;
      }
      sub_C43C90(&v34, a3, v35);
      v17 = v35;
    }
    if ( v17 > 0x40 )
    {
      sub_C43B90(&v34, &v23);
      v17 = v35;
      v19 = v34;
      goto LABEL_40;
    }
    v18 = v34;
LABEL_39:
    v19 = v23 & v18;
LABEL_40:
    v29 = v17;
    v28 = v19;
    sub_C46B40(&v23, &v28);
    sub_C46B40(&s, &v28);
    sub_969240(&v28);
    goto LABEL_9;
  }
  sub_AADB10(a1, a3, 0);
  return a1;
}
