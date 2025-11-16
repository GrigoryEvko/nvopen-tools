// Function: sub_9F1EB0
// Address: 0x9f1eb0
//
__int64 __fastcall sub_9F1EB0(__int64 a1, const __m128i *a2, unsigned __int64 a3, char a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v12; // rax
  char v13; // al
  _QWORD v14[2]; // [rsp+0h] [rbp-70h] BYREF
  char v15; // [rsp+10h] [rbp-60h]
  const __m128i *v16; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v17; // [rsp+28h] [rbp-48h]
  __int16 v18; // [rsp+40h] [rbp-30h]

  v16 = a2;
  v17 = a3;
  v18 = 261;
  sub_C7EAD0(v14, &v16, 0, 1, 0);
  if ( (v15 & 1) != 0 )
  {
    sub_C63CA0(&v16, LODWORD(v14[0]), v14[1]);
    v12 = (unsigned __int64)v16;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v12 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v15 & 1) != 0 )
      return a1;
  }
  else
  {
    v6 = v14[0];
    if ( a4 && *(_QWORD *)(v14[0] + 16LL) == *(_QWORD *)(v14[0] + 8LL) )
    {
      v13 = *(_BYTE *)(a1 + 8);
      *(_QWORD *)a1 = 0;
      *(_BYTE *)(a1 + 8) = v13 & 0xFC | 2;
      goto LABEL_10;
    }
    *(double *)a5.m128i_i64 = sub_C7EC60(&v16, v14[0]);
    sub_9F1E00(a1, v6, v7, v8, v9, v10, a5, v16, v17);
    if ( (v15 & 1) != 0 )
      return a1;
  }
  v6 = v14[0];
  if ( !v14[0] )
    return a1;
LABEL_10:
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  return a1;
}
