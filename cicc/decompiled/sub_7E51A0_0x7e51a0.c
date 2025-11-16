// Function: sub_7E51A0
// Address: 0x7e51a0
//
__int64 __fastcall sub_7E51A0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 result; // rax
  char *v3; // r13
  __int64 v4; // rax
  __m128i *v5; // rax
  _BYTE *v6; // r13
  int v7; // edi
  char *v8; // r13
  __int64 v9; // rax
  __m128i *v10; // rax
  int v11[9]; // [rsp+Ch] [rbp-24h] BYREF

  v1 = *(_QWORD *)(a1 + 256);
  if ( v1 )
  {
    result = *(_QWORD *)(v1 + 16);
    if ( result )
      return result;
  }
  else
  {
    v1 = sub_726210(a1);
    result = *(_QWORD *)(v1 + 16);
    if ( result )
      return result;
  }
  sub_7296C0(v11);
  if ( *(_BYTE *)(a1 + 172) == 2 )
  {
    v8 = (char *)sub_815620("__IFV__");
    v9 = sub_72D2E0(*(_QWORD **)(a1 + 152));
    v10 = sub_7E2190(v8, 1, v9, 2);
    *(_QWORD *)(v1 + 16) = v10;
    v10[5].m128i_i8[9] |= 8u;
  }
  else
  {
    v3 = (char *)sub_815620("__IFV__");
    v4 = sub_72D2E0(*(_QWORD **)(a1 + 152));
    v5 = sub_7E2190(v3, 1, v4, 0);
    *(_QWORD *)(v1 + 16) = v5;
    v5[5].m128i_i8[9] |= 8u;
    sub_7E4C10(*(_QWORD *)(v1 + 16));
  }
  v6 = sub_724D50(6);
  sub_72D3B0(a1, (__int64)v6, 1);
  v7 = v11[0];
  *(_QWORD *)(*(_QWORD *)(v1 + 16) + 184LL) = v6;
  *(_BYTE *)(*(_QWORD *)(v1 + 16) + 177LL) = 1;
  sub_729730(v7);
  return *(_QWORD *)(v1 + 16);
}
