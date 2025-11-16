// Function: sub_2F1BF70
// Address: 0x2f1bf70
//
__int64 __fastcall sub_2F1BF70(__int64 a1, unsigned __int64 *a2)
{
  int v3; // ebx
  __int64 v4; // rax
  unsigned __int64 v5; // r15
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  char v11; // al
  _BOOL8 v12; // rcx
  size_t v13; // rdx
  void **p_s2; // rsi
  unsigned __int64 *v16; // r15
  unsigned __int64 *v17; // rbx
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // [rsp+8h] [rbp-A8h]
  __int64 v20; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v22; // [rsp+18h] [rbp-98h]
  __int64 v23; // [rsp+20h] [rbp-90h]
  unsigned __int64 v24; // [rsp+20h] [rbp-90h]
  __int64 v25; // [rsp+28h] [rbp-88h]
  char v26; // [rsp+3Fh] [rbp-71h] BYREF
  __int64 v27; // [rsp+40h] [rbp-70h] BYREF
  void **v28; // [rsp+48h] [rbp-68h] BYREF
  void *s2; // [rsp+50h] [rbp-60h] BYREF
  __int64 v30; // [rsp+58h] [rbp-58h]
  _QWORD v31[2]; // [rsp+60h] [rbp-50h] BYREF
  __m128i v32[4]; // [rsp+70h] [rbp-40h] BYREF

  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = -1431655765 * ((__int64)(a2[1] - *a2) >> 5);
  if ( v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 1;
    v6 = 0;
    v25 = v4 + 2;
    do
    {
      while ( 1 )
      {
        v7 = v6 + 96;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
               a1,
               (unsigned int)(v5 - 1),
               &v27) )
        {
          break;
        }
        v6 += 96;
        if ( v25 == ++v5 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      v8 = *a2;
      v9 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a2[1] - *a2) >> 5);
      if ( v5 - 1 >= v9 )
      {
        if ( v9 < v5 )
        {
          sub_2F1BC60(a2, v5 - v9);
          v8 = *a2;
        }
        else if ( v9 > v5 )
        {
          v24 = v8 + v7;
          if ( a2[1] != v8 + v7 )
          {
            v22 = v5;
            v16 = (unsigned __int64 *)a2[1];
            v20 = v6;
            v17 = (unsigned __int64 *)(v8 + v7);
            do
            {
              v18 = v17[6];
              if ( (unsigned __int64 *)v18 != v17 + 8 )
                j_j___libc_free_0(v18);
              if ( (unsigned __int64 *)*v17 != v17 + 2 )
                j_j___libc_free_0(*v17);
              v17 += 12;
            }
            while ( v16 != v17 );
            v5 = v22;
            v6 = v20;
            a2[1] = v24;
            v8 = *a2;
          }
        }
      }
      v23 = v8 + v6;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 144LL))(a1);
      if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, void ***, void **))(*(_QWORD *)a1 + 120LL))(
             a1,
             "reg",
             1,
             0,
             &v28,
             &s2) )
      {
        sub_2F0E9C0(a1, v23);
        (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
      }
      LOBYTE(v31[0]) = 0;
      s2 = v31;
      v30 = 0;
      v10 = *(_QWORD *)a1;
      v32[0] = 0;
      v19 = (unsigned __int64 *)(v23 + 48);
      v11 = (*(__int64 (__fastcall **)(__int64))(v10 + 16))(a1);
      v12 = 0;
      if ( v11 )
      {
        v13 = *(_QWORD *)(v23 + 56);
        if ( v13 == v30 )
        {
          v12 = 1;
          if ( v13 )
            v12 = memcmp(*(const void **)(v23 + 48), s2, v13) == 0;
        }
      }
      p_s2 = (void **)"virtual-reg";
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, void ***))(*(_QWORD *)a1 + 120LL))(
             a1,
             "virtual-reg",
             0,
             v12,
             &v26,
             &v28) )
      {
        sub_2F0E9C0(a1, (__int64)v19);
        p_s2 = v28;
        (*(void (__fastcall **)(__int64, void **))(*(_QWORD *)a1 + 128LL))(a1, v28);
      }
      else if ( v26 )
      {
        p_s2 = &s2;
        sub_2240AE0(v19, (unsigned __int64 *)&s2);
        *(__m128i *)(v23 + 80) = _mm_loadu_si128(v32);
      }
      if ( s2 != v31 )
      {
        p_s2 = (void **)(v31[0] + 1LL);
        j_j___libc_free_0((unsigned __int64)s2);
      }
      v6 = v7;
      ++v5;
      (*(void (__fastcall **)(__int64, void **))(*(_QWORD *)a1 + 152LL))(a1, p_s2);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v27);
    }
    while ( v25 != v5 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
