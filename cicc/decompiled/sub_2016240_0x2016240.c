// Function: sub_2016240
// Address: 0x2016240
//
__int64 __fastcall sub_2016240(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned __int8 a3,
        __int64 a4,
        unsigned int a5,
        int a6,
        __int64 a7)
{
  unsigned int v7; // r12d
  __int64 v8; // rax
  _QWORD *v10; // rdi
  __int64 v12; // rcx
  __int64 v13; // rax
  const __m128i *v14; // r9
  _BYTE *v15; // rdi
  unsigned int v16; // r15d
  __int64 v17; // rdx
  __int64 v18; // r13
  _BYTE *v19; // rbx
  const __m128i *v21; // r9
  _BYTE *v23; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v24; // [rsp+18h] [rbp-B8h]
  _BYTE v25[176]; // [rsp+20h] [rbp-B0h] BYREF

  v7 = 0;
  v8 = *(unsigned __int16 *)(a2 + 24);
  if ( !a3 )
    return v7;
  v10 = (_QWORD *)*a1;
  if ( (unsigned int)v8 <= 0x102 && *((_BYTE *)v10 + 259 * a3 + v8 + 2422) != 4 )
    return v7;
  v12 = a1[1];
  v23 = v25;
  v24 = 0x800000000LL;
  v13 = *v10;
  if ( (_BYTE)a5 )
  {
    (*(void (__fastcall **)(_QWORD *, unsigned __int64, _BYTE **, __int64))(v13 + 1320))(v10, a2, &v23, v12);
    v7 = v24;
    if ( (_DWORD)v24 )
    {
      if ( *(_DWORD *)(a2 + 60) + 1 == (_DWORD)v24 )
      {
        sub_2015C40(
          (__int64)a1,
          a2,
          0,
          *(_QWORD *)v23,
          *((__m128i **)v23 + 1),
          v14,
          *((_QWORD *)v23 + 2),
          *((_QWORD *)v23 + 3));
        if ( *(_DWORD *)(a2 + 60) > 1u )
          sub_2013400((__int64)a1, a2, 1, *((_QWORD *)v23 + 4), *((__m128i **)v23 + 5), v21);
        v15 = v23;
        v7 = a5;
        goto LABEL_11;
      }
      goto LABEL_7;
    }
  }
  else
  {
    (*(void (__fastcall **)(_QWORD *, unsigned __int64, _BYTE **, __int64))(v13 + 1304))(v10, a2, &v23, v12);
    v7 = v24;
    if ( (_DWORD)v24 )
    {
LABEL_7:
      v15 = v23;
      v16 = 0;
      do
      {
        v17 = v16++;
        v18 = 16 * v17;
        sub_2013400((__int64)a1, a2, v17, *(_QWORD *)&v15[16 * v17], *(__m128i **)&v15[16 * v17 + 8], v14);
        v15 = v23;
        *(_DWORD *)(*(_QWORD *)&v23[v18] + 64LL) = *(_DWORD *)(a2 + 64);
      }
      while ( v16 != v7 );
      v7 = 1;
      if ( a7 )
      {
        v19 = &v15[16 * a6];
        *(_QWORD *)a7 = *(_QWORD *)v19;
        *(_DWORD *)(a7 + 8) = *((_DWORD *)v19 + 2);
      }
      goto LABEL_11;
    }
  }
  v15 = v23;
LABEL_11:
  if ( v15 != v25 )
    _libc_free((unsigned __int64)v15);
  return v7;
}
