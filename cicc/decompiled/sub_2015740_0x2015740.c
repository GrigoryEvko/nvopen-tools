// Function: sub_2015740
// Address: 0x2015740
//
__int64 __fastcall sub_2015740(__int64 *a1, unsigned __int64 a2, unsigned __int8 a3)
{
  unsigned int v3; // r13d
  __int64 v4; // rax
  __int64 v6; // rdi
  __int64 v7; // rcx
  const __m128i *v8; // r9
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 *v11; // rax
  __int64 v12; // rcx
  __m128i *v13; // r8
  _BYTE *v15; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+8h] [rbp-B8h]
  _BYTE v17[176]; // [rsp+10h] [rbp-B0h] BYREF

  v3 = 0;
  v4 = *(unsigned __int16 *)(a2 + 24);
  if ( a3 )
  {
    v6 = *a1;
    if ( (unsigned int)v4 > 0x102 || *(_BYTE *)(v4 + v6 + 259LL * a3 + 2422) == 4 )
    {
      v7 = a1[1];
      v3 = 0;
      v16 = 0x800000000LL;
      v15 = v17;
      (*(void (__fastcall **)(__int64, unsigned __int64, _BYTE **, __int64))(*(_QWORD *)v6 + 1320LL))(v6, a2, &v15, v7);
      v9 = (unsigned int)v16;
      if ( (_DWORD)v16 )
      {
        v10 = 0;
        do
        {
          while ( 1 )
          {
            v11 = (__int64 *)&v15[16 * v10];
            v12 = *v11;
            v13 = (__m128i *)v11[1];
            if ( *(_BYTE *)(*(_QWORD *)(*v11 + 40) + 16LL * *((unsigned int *)v11 + 2)) == 1 )
              break;
            sub_2015400((__int64)a1, a2, (unsigned int)v10++, v12, v13, v8);
            if ( v9 == v10 )
              goto LABEL_9;
          }
          sub_2013400((__int64)a1, a2, (unsigned int)v10++, v12, v13, v8);
        }
        while ( v9 != v10 );
LABEL_9:
        v3 = 1;
      }
      if ( v15 != v17 )
        _libc_free((unsigned __int64)v15);
    }
  }
  return v3;
}
