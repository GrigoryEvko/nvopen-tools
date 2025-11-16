// Function: sub_3701060
// Address: 0x3701060
//
unsigned __int64 *__fastcall sub_3701060(unsigned __int64 *a1, _QWORD *a2, _QWORD *a3, const __m128i *a4)
{
  __int64 v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  __int64 v12; // rax
  _OWORD v14[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v15; // [rsp+30h] [rbp-30h]

  v6 = a2[7];
  v7 = a2[5];
  if ( v6 )
  {
    if ( !v7 && !a2[6] )
    {
      if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v6 + 40LL))(v6) )
      {
        v12 = a4[2].m128i_i64[0];
        v14[0] = _mm_loadu_si128(a4);
        v15 = v12;
        v14[1] = _mm_loadu_si128(a4 + 1);
        if ( (unsigned __int8)v12 > 1u )
          (*(void (__fastcall **)(_QWORD, _OWORD *))(*(_QWORD *)a2[7] + 24LL))(a2[7], v14);
      }
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(*(_QWORD *)a2[7] + 16LL))(a2[7], *a3, a3[1]);
      if ( a2[7] && !a2[5] && !a2[6] )
        a2[8] += a3[1];
      goto LABEL_7;
    }
LABEL_3:
    if ( *(_BYTE *)(v7 + 48) )
    {
      v9 = *(_QWORD *)(v7 + 40);
    }
    else
    {
      v8 = *(_QWORD *)(v7 + 24);
      LODWORD(v9) = 0;
      if ( v8 )
        LODWORD(v9) = (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD *, _QWORD))(*(_QWORD *)v8 + 40LL))(
                        v8,
                        a2,
                        a3,
                        0)
                    - *(_DWORD *)(v7 + 32);
    }
    sub_1254950((unsigned __int64 *)v14, v7, (__int64)a3, v9 - *(_DWORD *)(v7 + 56));
    v10 = *(_QWORD *)&v14[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (*(_QWORD *)&v14[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
      goto LABEL_7;
    goto LABEL_20;
  }
  a2 = (_QWORD *)a2[6];
  if ( !a2 || v7 )
    goto LABEL_3;
  sub_3719260(v14, a2, *a3, a3[1]);
  v10 = *(_QWORD *)&v14[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (*(_QWORD *)&v14[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
LABEL_7:
    *a1 = 1;
    return a1;
  }
LABEL_20:
  *a1 = v10 | 1;
  return a1;
}
