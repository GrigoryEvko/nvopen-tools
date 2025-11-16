// Function: sub_6FEAC0
// Address: 0x6feac0
//
__int64 __fastcall sub_6FEAC0(__int64 a1, const __m128i *a2, __int64 a3, __m128i *a4, __int64 *a5, __int64 a6)
{
  int v8; // r13d
  __int64 v10; // rax
  bool v11; // zf
  __int64 v12; // rax
  __int64 i; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  _DWORD *v23; // rdx
  int v24; // esi
  unsigned int v26; // [rsp+10h] [rbp-1A0h] BYREF
  unsigned int v27; // [rsp+14h] [rbp-19Ch] BYREF
  __int64 v28; // [rsp+18h] [rbp-198h] BYREF
  _DWORD v29[36]; // [rsp+20h] [rbp-190h] BYREF
  __int64 v30; // [rsp+B0h] [rbp-100h]

  v8 = a6;
  v10 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v11 = a2[1].m128i_i8[0] == 0;
  v28 = v10;
  if ( v11 )
    goto LABEL_5;
  v12 = a2->m128i_i64[0];
  for ( i = *(unsigned __int8 *)(a2->m128i_i64[0] + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v12 + 140) )
    v12 = *(_QWORD *)(v12 + 160);
  if ( !(_BYTE)i )
    goto LABEL_5;
  v26 = 1;
  if ( (unsigned int)sub_82ED00(a2, a2, i) || (unsigned int)sub_8DBE70(a3) )
  {
    v16 = v26;
    v27 = 1;
  }
  else
  {
    v27 = 0;
    if ( !(_BYTE)a1 )
      goto LABEL_16;
    sub_6E6B60(a2, 0, v18, v19, v15, v16);
    if ( a2[1].m128i_i8[0] == 2 )
    {
      v23 = 0;
      v29[0] = 0;
      v24 = (_DWORD)a2 + 144;
      if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
        v23 = v29;
      sub_712770(
        (unsigned __int8)a1,
        v24,
        a3,
        v28,
        *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u,
        *(_BYTE *)(qword_4D03C50 + 17LL) & 1,
        (__int64)&v26,
        (__int64)&v27,
        (__int64)v23,
        (__int64)a5);
      v15 = v29[0];
      if ( v29[0] )
        sub_6E50A0();
    }
    if ( !v26 )
    {
      if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
      {
        sub_6F7B30((__int64)a2, a1, a3, (__int64)v29, v15, v16);
        v17 = v28;
        *(_QWORD *)(v28 + 144) = v30;
      }
      else
      {
        v17 = v28;
      }
      sub_6E6A50(v17, (__int64)a4);
      goto LABEL_6;
    }
    if ( !v27 )
    {
LABEL_16:
      if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u && (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 && !word_4D04898 )
      {
        if ( (unsigned int)sub_6E5430() )
          sub_6851C0(0x1Cu, a5);
LABEL_5:
        sub_6E6260(a4);
        goto LABEL_6;
      }
    }
  }
  sub_6F7B30((__int64)a2, a1, a3, (__int64)a4, v15, v16);
  if ( v27 )
    sub_6F4B70(a4, (unsigned __int8)a1, v27, v20, v21, v22);
LABEL_6:
  a4[4].m128i_i8[0] = a2[4].m128i_i8[0];
  sub_6E3BA0((__int64)a4, a5, v8, 0);
  return sub_724E30(&v28);
}
