// Function: sub_1365EE0
// Address: 0x1365ee0
//
_QWORD *__fastcall sub_1365EE0(__int64 *a1, __int64 a2, unsigned int a3)
{
  _QWORD *result; // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r13
  unsigned __int64 v11; // r14
  __int64 v12; // rdx
  _BOOL4 v13; // r10d
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdi
  __int64 *v18; // rax
  __int64 v19; // rcx
  bool v20; // zf
  __int64 v21; // [rsp+8h] [rbp-98h]
  _BOOL4 v23; // [rsp+1Ch] [rbp-84h]
  __int64 v24; // [rsp+20h] [rbp-80h]
  unsigned int v25; // [rsp+34h] [rbp-6Ch] BYREF
  __int64 v26; // [rsp+38h] [rbp-68h] BYREF
  __m128i v27; // [rsp+40h] [rbp-60h] BYREF
  __int64 v28; // [rsp+50h] [rbp-50h]
  __int64 v29; // [rsp+58h] [rbp-48h]
  __int64 v30; // [rsp+60h] [rbp-40h]

  result = (_QWORD *)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (_DWORD)result )
  {
    v6 = 0;
    v21 = 24LL * (unsigned int)result;
    do
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v7 = *(_QWORD *)(a2 - 8);
      else
        v7 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v8 = sub_14AD280(*(_QWORD *)(v7 + v6), *(_QWORD *)(a1[2] + 8), 6);
      v9 = sub_1CCAE90(v8, 1);
      v10 = a1[4];
      v26 = v9;
      v11 = v9;
      result = sub_1361E90(v10, (unsigned __int64 *)&v26);
      if ( v12 )
      {
        v13 = 1;
        if ( !result && v12 != v10 + 8 )
          v13 = v11 < *(_QWORD *)(v12 + 32);
        v23 = v13;
        v24 = v12;
        v14 = sub_22077B0(40);
        *(_QWORD *)(v14 + 32) = v26;
        sub_220F040(v23, v14, v24, v10 + 8);
        v16 = v26;
        ++*(_QWORD *)(v10 + 40);
        if ( *(_BYTE *)(v16 + 16) == 77 )
        {
          v19 = *a1;
          v27.m128i_i64[0] = v16;
          v20 = *(_QWORD *)(v19 + 16) == 0;
          v25 = a3;
          if ( v20 )
            sub_4263D6(v23, v14, v15);
          (*(void (__fastcall **)(__int64, __m128i *, unsigned int *))(v19 + 24))(v19, &v27, &v25);
        }
        else
        {
          v27.m128i_i64[0] = v16;
          v28 = 0;
          v17 = (__int64 *)a1[2];
          v27.m128i_i64[1] = a3;
          v18 = (__int64 *)a1[3];
          v29 = 0;
          v30 = 0;
          *(_BYTE *)a1[1] |= sub_1365330(v17, *v18, &v27);
        }
        result = (_QWORD *)a1[1];
        if ( *(_BYTE *)result == 7 )
          break;
      }
      v6 += 24;
    }
    while ( v21 != v6 );
  }
  return result;
}
