// Function: sub_2F1EEA0
// Address: 0x2f1eea0
//
__int64 __fastcall sub_2F1EEA0(__int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // r15
  int v3; // ebx
  __int64 v4; // rax
  unsigned __int64 v5; // r12
  __int64 v6; // rbx
  char i; // al
  unsigned __int64 v9; // r15
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  char v14; // al
  _BOOL8 v15; // rcx
  size_t v16; // rdx
  void **p_s2; // rsi
  char v18; // al
  __int64 v19; // rcx
  char v20; // al
  __int64 v21; // rcx
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // rdi
  unsigned __int64 *v26; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v27; // [rsp+10h] [rbp-A0h]
  __int64 v28; // [rsp+20h] [rbp-90h]
  __int64 v29; // [rsp+28h] [rbp-88h]
  char v30; // [rsp+3Fh] [rbp-71h] BYREF
  __int64 v31; // [rsp+40h] [rbp-70h] BYREF
  void **v32; // [rsp+48h] [rbp-68h] BYREF
  void *s2; // [rsp+50h] [rbp-60h] BYREF
  __int64 v34; // [rsp+58h] [rbp-58h]
  _QWORD v35[2]; // [rsp+60h] [rbp-50h] BYREF
  __m128i v36[4]; // [rsp+70h] [rbp-40h] BYREF

  v2 = a1;
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = -858993459 * ((__int64)(a2[1] - *a2) >> 4);
  if ( v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 1;
    v6 = 0;
    v28 = v4 + 2;
    for ( i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(a1, 0, &v31);
          ;
          i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
                a1,
                (unsigned int)(v5 - 1),
                &v31) )
    {
      v29 = v6 + 80;
      if ( i )
      {
        v9 = a2[1];
        v10 = *a2;
        v11 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v9 - *a2) >> 4);
        if ( v11 <= v5 - 1 )
        {
          if ( v11 < v5 )
          {
            sub_2F1EBD0(a2, v5 - v11);
            v10 = *a2;
          }
          else if ( v11 > v5 )
          {
            v27 = v10 + v6 + 80;
            if ( v9 != v27 )
            {
              v23 = v10 + v6 + 80;
              do
              {
                v24 = *(_QWORD *)(v23 + 24);
                if ( v24 != v23 + 40 )
                  j_j___libc_free_0(v24);
                v23 += 80LL;
              }
              while ( v9 != v23 );
              v10 = *a2;
              a2[1] = v27;
            }
          }
        }
        v12 = v10 + v6;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, void ***, void **))(*(_QWORD *)a1 + 120LL))(
               a1,
               "id",
               1,
               0,
               &v32,
               &s2) )
        {
          sub_2F08170(a1, v12);
          (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
        }
        LOBYTE(v35[0]) = 0;
        s2 = v35;
        v26 = (unsigned __int64 *)(v12 + 24);
        v13 = *(_QWORD *)a1;
        v34 = 0;
        v36[0] = 0;
        v14 = (*(__int64 (__fastcall **)(__int64))(v13 + 16))(a1);
        v15 = 0;
        if ( v14 )
        {
          v16 = *(_QWORD *)(v12 + 32);
          if ( v16 == v34 )
          {
            v15 = 1;
            if ( v16 )
              v15 = memcmp(*(const void **)(v12 + 24), s2, v16) == 0;
          }
        }
        p_s2 = (void **)"value";
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, char *, void ***))(*(_QWORD *)a1 + 120LL))(
               a1,
               "value",
               0,
               v15,
               &v30,
               &v32) )
        {
          sub_2F0E9C0(a1, (__int64)v26);
          p_s2 = v32;
          (*(void (__fastcall **)(__int64, void **))(*(_QWORD *)a1 + 128LL))(a1, v32);
        }
        else if ( v30 )
        {
          p_s2 = &s2;
          sub_2240AE0(v26, (unsigned __int64 *)&s2);
          *(__m128i *)(v12 + 56) = _mm_loadu_si128(v36);
        }
        if ( s2 != v35 )
        {
          p_s2 = (void **)(v35[0] + 1LL);
          j_j___libc_free_0((unsigned __int64)s2);
        }
        v18 = (*(__int64 (__fastcall **)(__int64, void **))(*(_QWORD *)a1 + 16LL))(a1, p_s2);
        v19 = 0;
        if ( v18 )
          v19 = *(_BYTE *)(v12 + 73) ^ 1u;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, __int64, void ***, void **))(*(_QWORD *)a1 + 120LL))(
               a1,
               "alignment",
               0,
               v19,
               &v32,
               &s2) )
        {
          sub_2F085F0(a1, (_BYTE *)(v12 + 72));
          (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
        }
        else if ( (_BYTE)v32 )
        {
          *(_BYTE *)(v12 + 73) = 0;
        }
        v20 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
        v21 = 0;
        if ( v20 )
          v21 = *(_BYTE *)(v12 + 74) ^ 1u;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, void ***, void **))(*(_QWORD *)a1 + 120LL))(
               a1,
               "isTargetSpecific",
               0,
               v21,
               &v32,
               &s2) )
        {
          sub_2F07940(a1, (_BYTE *)(v12 + 74));
          (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
        }
        else if ( (_BYTE)v32 )
        {
          *(_BYTE *)(v12 + 74) = 0;
        }
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v31);
      }
      v6 = v29;
      if ( v28 == ++v5 )
        break;
    }
    v2 = a1;
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2);
}
