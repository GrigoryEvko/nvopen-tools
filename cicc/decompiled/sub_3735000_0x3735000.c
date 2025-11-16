// Function: sub_3735000
// Address: 0x3735000
//
__int64 __fastcall sub_3735000(__int64 *a1, char a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v6; // dl
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  const char *v11; // [rsp+0h] [rbp-40h] BYREF
  char v12; // [rsp+20h] [rbp-20h]
  char v13; // [rsp+21h] [rbp-1Fh]

  if ( a1[51] )
    goto LABEL_2;
  v8 = a1[26];
  if ( !*(_BYTE *)(v8 + 3689) )
  {
    v9 = a1[23];
    v13 = 1;
    v11 = "cu_begin";
    v12 = 3;
    v10 = sub_31DCC50(v9, (__int64 *)&v11, a3, a4, a5);
    a1[24] = v10;
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1[23] + 224) + 208LL))(
      *(_QWORD *)(a1[23] + 224),
      v10,
      0);
    if ( a1[51] )
    {
LABEL_2:
      v6 = 5;
      goto LABEL_3;
    }
    v8 = a1[26];
  }
  if ( !*(_BYTE *)(v8 + 3769) )
  {
    sub_3248790(a1, a2, 1u);
    return sub_3220AA0(a1[26]);
  }
  v6 = 4;
LABEL_3:
  sub_3248790(a1, a2, v6);
  result = sub_3220AA0(a1[26]);
  if ( (unsigned __int16)result > 4u )
    return sub_31DCA30(a1[23], a1[92]);
  return result;
}
