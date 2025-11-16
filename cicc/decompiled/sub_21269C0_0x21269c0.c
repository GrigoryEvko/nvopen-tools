// Function: sub_21269C0
// Address: 0x21269c0
//
__int64 __fastcall sub_21269C0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  char *v9; // rdx
  __int64 v10; // r12
  const void **v11; // r8
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  unsigned int v14; // r15d
  __int64 v16; // rsi
  __int64 *v17; // r10
  __int128 v18; // [rsp-10h] [rbp-60h]
  __int64 *v19; // [rsp+0h] [rbp-50h]
  const void **v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h] BYREF
  int v22; // [rsp+18h] [rbp-38h]

  v6 = sub_2125740(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = v7;
  v9 = *(char **)(a2 + 40);
  v10 = v6;
  v11 = (const void **)*((_QWORD *)v9 + 1);
  v12 = *v9;
  v13 = *(_QWORD *)(v10 + 40);
  v14 = v12;
  if ( *(_BYTE *)v13 != v12 || *(const void ***)(v13 + 8) != v11 && !v12 )
  {
    v16 = *(_QWORD *)(a2 + 72);
    v17 = *(__int64 **)(a1 + 8);
    v21 = v16;
    if ( v16 )
    {
      v19 = v17;
      v20 = v11;
      sub_1623A60((__int64)&v21, v16, 2);
      v17 = v19;
      v11 = v20;
    }
    *((_QWORD *)&v18 + 1) = v8;
    *(_QWORD *)&v18 = v10;
    v22 = *(_DWORD *)(a2 + 64);
    v10 = sub_1D309E0(v17, 157, (__int64)&v21, v14, v11, 0, a3, a4, a5, v18);
    if ( v21 )
      sub_161E7C0((__int64)&v21, v21);
  }
  return v10;
}
