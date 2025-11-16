// Function: sub_7F36F0
// Address: 0x7f36f0
//
void __fastcall sub_7F36F0(_QWORD *a1)
{
  __m128i *v1; // r13
  __m128i *v2; // r14
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __m128i *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // r8
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  int v15; // [rsp+Ch] [rbp-104h] BYREF
  _QWORD v16[10]; // [rsp+10h] [rbp-100h] BYREF
  unsigned int v17; // [rsp+60h] [rbp-B0h]

  v1 = (__m128i *)a1[9];
  v2 = (__m128i *)v1[1].m128i_i64[0];
  if ( dword_4F077C4 == 2 )
    sub_7F2A70(v1, 0);
  else
    sub_7D98E0((__int64)v1, 0);
  v3 = unk_4F06964;
  if ( !sub_7E6F30((__int64)v1, unk_4F06964, &v15)
    || (unsigned int)sub_731D60((__int64)v2, v3, v4, v5, v6, v7)
    || dword_4F077C4 == 2
    && (sub_76C7C0((__int64)v16), v16[0] = sub_7E0550, sub_76CDC0(v2, (__int64)v16, v12, v13, v14), (v9 = v17) != 0) )
  {
    v8 = v2;
    if ( dword_4F077C4 != 2 )
    {
LABEL_5:
      sub_7D98E0((__int64)v8, 0);
      return;
    }
    goto LABEL_12;
  }
  v11 = *((_BYTE *)a1 + 56);
  if ( v11 == 87 )
  {
    if ( !v15 )
      goto LABEL_14;
LABEL_11:
    v8 = v2;
    if ( dword_4F077C4 != 2 )
      goto LABEL_5;
LABEL_12:
    sub_7F2A70(v8, 0);
    return;
  }
  if ( v11 != 88 || !v15 )
    goto LABEL_11;
LABEL_14:
  if ( *a1 == v1->m128i_i64[0] || (unsigned int)sub_8D97D0(*a1, v1->m128i_i64[0], 1, v9, v10) )
  {
    sub_730620((__int64)a1, v1);
  }
  else
  {
    sub_7E2300((__int64)a1, (__int64)v1, *a1);
    v1[1].m128i_i64[0] = 0;
  }
}
