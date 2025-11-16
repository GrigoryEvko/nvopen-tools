// Function: sub_7E2E30
// Address: 0x7e2e30
//
__int64 __fastcall sub_7E2E30(__int64 a1, __int64 a2, int a3, __int64 *a4, __int64 *a5, int a6, __int64 a7)
{
  __int64 v10; // rbx
  _QWORD *v11; // rbx
  __m128i *v12; // r15
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 result; // rax
  const __m128i *v20; // r15
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  _QWORD *v29; // [rsp+8h] [rbp-38h]

  if ( a2 )
  {
    v10 = a2;
    if ( (*(_QWORD *)(a2 + 192) & 0x10000000000008LL) == 0x10000000000000LL )
    {
      v29 = sub_7E1330();
      v20 = (const __m128i *)sub_724D50(6);
      if ( (*(_BYTE *)(a2 + 192) & 8) != 0 )
      {
        sub_72CBE0();
        v10 = sub_7F8110("__cxa_pure_virtual", 0, 0, 0, 0);
      }
      else if ( (*(_BYTE *)(a2 + 206) & 0x10) != 0 )
      {
        sub_72CBE0();
        v10 = sub_7F8110("__cxa_deleted_virtual", 0, 0, 0, 0);
      }
      sub_72D3B0(v10, (__int64)v20, 1);
      sub_70FEE0((__int64)v20, (__int64)v29, v25, v26, v27);
      *(_BYTE *)(v10 + 88) |= 4u;
      goto LABEL_16;
    }
  }
  v11 = sub_7E1330();
  v12 = (__m128i *)sub_724D50(1);
  sub_7E2DB0(v12, a1, unk_4F06895, a7, 0);
  v15 = (_QWORD *)v12[8].m128i_i64[0];
  if ( v15 != v11 && !(unsigned int)sub_8D97D0(v15, v11, 1, v13, v14) )
    sub_70FEE0((__int64)v12, (__int64)v11, v16, v17, v18);
  result = *a4;
  if ( *a4 )
  {
    if ( a6 )
    {
      v12[7].m128i_i64[1] = result;
      *a4 = (__int64)v12;
    }
    else
    {
      result = *a5;
      *(_QWORD *)(*a5 + 120) = v12;
      *a5 = (__int64)v12;
    }
  }
  else
  {
    *a5 = (__int64)v12;
    *a4 = (__int64)v12;
  }
  if ( a3 )
  {
    v20 = (const __m128i *)sub_724D50(6);
    if ( dword_4D04848 )
    {
      v21 = sub_7DC650(a7);
      sub_72D510(v21, (__int64)v20, 1);
      sub_70FEE0((__int64)v20, (__int64)v11, v22, v23, v24);
    }
    else
    {
      sub_72BB40((__int64)v11, v20);
    }
LABEL_16:
    result = *a4;
    if ( *a4 )
    {
      if ( a6 )
      {
        v20[7].m128i_i64[1] = result;
        *a4 = (__int64)v20;
      }
      else
      {
        result = *a5;
        *(_QWORD *)(*a5 + 120) = v20;
        *a5 = (__int64)v20;
      }
    }
    else
    {
      *a5 = (__int64)v20;
      *a4 = (__int64)v20;
    }
  }
  return result;
}
