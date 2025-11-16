// Function: sub_15E33D0
// Address: 0x15e33d0
//
__int64 __fastcall sub_15E33D0(__int64 a1, __int64 a2)
{
  int v2; // r13d
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  const void *v10; // r15
  _QWORD *v11; // r8
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // [rsp+8h] [rbp-E8h]
  __int64 *v15; // [rsp+28h] [rbp-C8h]
  _QWORD *v16; // [rsp+28h] [rbp-C8h]
  unsigned int *v17[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 *v18; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v19; // [rsp+48h] [rbp-A8h]
  _BYTE v20[32]; // [rsp+50h] [rbp-A0h] BYREF
  void *s2; // [rsp+70h] [rbp-80h] BYREF
  size_t n; // [rsp+78h] [rbp-78h]
  _QWORD v23[14]; // [rsp+80h] [rbp-70h] BYREF

  v2 = *(_DWORD *)(a2 + 36);
  if ( !v2 )
  {
    *(_BYTE *)(a1 + 8) = 0;
    return a1;
  }
  v4 = *(_QWORD *)(a2 + 24);
  v18 = (__int64 *)v20;
  v19 = 0x400000000LL;
  s2 = v23;
  n = 0x800000000LL;
  v14 = v4;
  sub_15E1220(v2, (__int64)&s2);
  v17[0] = (unsigned int *)s2;
  v17[1] = (unsigned int *)(unsigned int)n;
  if ( !sub_15E2EC0(**(_QWORD **)(v4 + 16), v17, (__int64 *)&v18) )
  {
    v5 = *(_QWORD *)(v4 + 16);
    v15 = (__int64 *)(v5 + 8LL * *(unsigned int *)(v4 + 12));
    if ( v15 == (__int64 *)(v5 + 8) )
    {
LABEL_13:
      if ( !(unsigned __int8)sub_15E3390(*(_DWORD *)(v14 + 8) >> 8 != 0, (__int64 *)v17) )
      {
        if ( s2 != v23 )
          _libc_free((unsigned __int64)s2);
        v7 = sub_1649960(a2);
        v9 = v8;
        v10 = (const void *)v7;
        sub_15E1070((__int64 *)&s2, v2, v18, (unsigned int)v19);
        v11 = s2;
        if ( n != v9 || n && (v16 = s2, v13 = memcmp(v10, s2, n), v11 = v16, v13) )
        {
          if ( v11 != v23 )
            j_j___libc_free_0(v11, v23[0] + 1LL);
          v12 = sub_15E26F0(*(__int64 **)(a2 + 40), v2, v18, (unsigned int)v19);
          *(_WORD *)(v12 + 18) = *(_WORD *)(v12 + 18) & 0xC00F | *(_WORD *)(a2 + 18) & 0x3FF0;
          *(_BYTE *)(a1 + 8) = 1;
          *(_QWORD *)a1 = v12;
        }
        else
        {
          if ( v11 != v23 )
            j_j___libc_free_0(v11, v23[0] + 1LL);
          *(_BYTE *)(a1 + 8) = 0;
        }
        goto LABEL_11;
      }
    }
    else
    {
      v6 = (__int64 *)(v5 + 8);
      while ( !sub_15E2EC0(*v6, v17, (__int64 *)&v18) )
      {
        if ( v15 == ++v6 )
          goto LABEL_13;
      }
    }
  }
  *(_BYTE *)(a1 + 8) = 0;
  if ( s2 != v23 )
    _libc_free((unsigned __int64)s2);
LABEL_11:
  if ( v18 != (__int64 *)v20 )
    _libc_free((unsigned __int64)v18);
  return a1;
}
