// Function: sub_E8AF40
// Address: 0xe8af40
//
__int64 __fastcall sub_E8AF40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 (*v7)(void); // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 result; // rax
  unsigned __int64 v11; // r13
  unsigned int v12; // r15d
  __int64 v13; // rbx
  _QWORD *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rdi
  unsigned int v20; // [rsp+Ch] [rbp-84h]
  __int64 v21; // [rsp+10h] [rbp-80h]
  __int64 v23; // [rsp+28h] [rbp-68h] BYREF
  const char *v24; // [rsp+30h] [rbp-60h] BYREF
  char v25; // [rsp+50h] [rbp-40h]
  char v26; // [rsp+51h] [rbp-3Fh]

  v7 = *(__int64 (**)(void))(*(_QWORD *)a1 + 80LL);
  if ( (char *)v7 == (char *)sub_E8A180 )
  {
    v8 = 0;
    if ( *(_BYTE *)(a1 + 276) )
      v8 = *(_QWORD *)(a1 + 296);
  }
  else
  {
    v8 = v7();
  }
  if ( sub_E81930(a2, &v23, v8) )
  {
    v9 = v23;
    if ( v23 < 0 )
    {
      v18 = *(_QWORD *)(a1 + 8);
      v26 = 1;
      v19 = *(__int64 **)(v18 + 80);
      v24 = "'.fill' directive with negative repeat count has no effect";
      v25 = 3;
      return (__int64)sub_C91CB0(v19, a5, 1, (__int64)&v24, 0, 0, 0, 0, 1u);
    }
    else
    {
      result = 4;
      if ( a3 <= 4 )
        result = a3;
      v21 = result;
      v11 = a4 & (0xFFFFFFFFFFFFFFFFLL >> (8 * (8 - (unsigned __int8)result)));
      if ( v23 )
      {
        v12 = result;
        v13 = 0;
        v20 = a3 - result;
        do
        {
          result = (*(__int64 (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)a1 + 536LL))(a1, v11, v12);
          if ( v21 < a3 )
            result = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a1 + 536LL))(a1, 0, v20);
          ++v13;
        }
        while ( v9 != v13 );
      }
    }
  }
  else
  {
    v14 = *(_QWORD **)(a1 + 8);
    v15 = v14[36];
    v14[46] += 56LL;
    v16 = (v15 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v14[37] >= (unsigned __int64)(v16 + 56) && v15 )
      v14[36] = v16 + 56;
    else
      v16 = sub_9D1E70((__int64)(v14 + 36), 56, 56, 3);
    sub_E81B30(v16, 2, 0);
    *(_QWORD *)(v16 + 32) = a4;
    *(_QWORD *)(v16 + 40) = a2;
    *(_BYTE *)(v16 + 30) = a3;
    *(_QWORD *)(v16 + 48) = a5;
    v17 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
    *(_QWORD *)(v16 + 8) = v17;
    *(_DWORD *)(v16 + 24) = *(_DWORD *)(*(_QWORD *)(a1 + 288) + 24LL) + 1;
    **(_QWORD **)(a1 + 288) = v16;
    *(_QWORD *)(a1 + 288) = v16;
    result = *(_QWORD *)(v17 + 8);
    *(_QWORD *)(result + 8) = v16;
  }
  return result;
}
