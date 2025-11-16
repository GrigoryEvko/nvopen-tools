// Function: sub_38D4FC0
// Address: 0x38d4fc0
//
__int64 __fastcall sub_38D4FC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 (*v8)(void); // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 result; // rax
  unsigned __int64 v12; // r13
  unsigned int v13; // r15d
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 *v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 *v23; // rdi
  unsigned int v24; // [rsp+Ch] [rbp-74h]
  __int64 v25; // [rsp+10h] [rbp-70h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  _QWORD *v28; // [rsp+18h] [rbp-68h]
  __int64 v29; // [rsp+28h] [rbp-58h] BYREF
  const char *v30; // [rsp+30h] [rbp-50h] BYREF
  char v31; // [rsp+40h] [rbp-40h]
  char v32; // [rsp+41h] [rbp-3Fh]

  v8 = *(__int64 (**)(void))(*(_QWORD *)a1 + 72LL);
  if ( (char *)v8 == (char *)sub_38D3BD0 )
  {
    LODWORD(v9) = 0;
    if ( *(_BYTE *)(a1 + 260) )
      v9 = *(_QWORD *)(a1 + 264);
  }
  else
  {
    LODWORD(v9) = v8();
  }
  if ( sub_38CF2B0(a2, &v29, v9) )
  {
    v10 = v29;
    if ( v29 < 0 )
    {
      v23 = **(__int64 ***)(a1 + 8);
      v32 = 1;
      v30 = "'.fill' directive with negative repeat count has no effect";
      v31 = 3;
      return (__int64)sub_16D14E0(v23, a5, 1, (__int64)&v30, 0, 0, 0, 0, 1u);
    }
    else
    {
      result = 4;
      if ( a3 <= 4 )
        result = a3;
      v25 = result;
      v12 = a4 & (0xFFFFFFFFFFFFFFFFLL >> (8 * (8 - (unsigned __int8)result)));
      if ( v29 )
      {
        v13 = result;
        v14 = 0;
        v24 = a3 - result;
        do
        {
          result = (*(__int64 (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)a1 + 424LL))(a1, v12, v13);
          if ( a3 > v25 )
            result = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a1 + 424LL))(a1, 0, v24);
          ++v14;
        }
        while ( v10 != v14 );
      }
    }
  }
  else
  {
    v15 = sub_38D4BB0(a1, 0);
    sub_38D4150(a1, v15, *(unsigned int *)(v15 + 72));
    v16 = sub_22077B0(0x50u);
    if ( v16 )
    {
      v26 = v16;
      v17 = v16;
      sub_38CF760(v16, 3, 0, 0);
      v16 = v26;
      *(_QWORD *)(v26 + 48) = a4;
      *(_BYTE *)(v26 + 56) = a3;
      *(_QWORD *)(v26 + 64) = a2;
      *(_QWORD *)(v26 + 72) = a5;
    }
    else
    {
      v17 = 0;
    }
    v28 = (_QWORD *)v16;
    sub_38D4150(a1, v16, 0);
    v18 = *(unsigned int *)(a1 + 120);
    v19 = 0;
    result = (__int64)v28;
    if ( (_DWORD)v18 )
      v19 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v18 - 32);
    v20 = *(__int64 **)(a1 + 272);
    v21 = *v20;
    v22 = *v28 & 7LL;
    v28[1] = v20;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    *v28 = v21 | v22;
    *(_QWORD *)(v21 + 8) = v17;
    *v20 = *v20 & 7 | v17;
    v28[3] = v19;
  }
  return result;
}
