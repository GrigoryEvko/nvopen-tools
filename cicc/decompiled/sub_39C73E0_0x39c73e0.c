// Function: sub_39C73E0
// Address: 0x39c73e0
//
__int64 __fastcall sub_39C73E0(__int64 *a1, char a2)
{
  unsigned __int8 v3; // dl
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  const char *v8; // [rsp+0h] [rbp-30h] BYREF
  char v9; // [rsp+10h] [rbp-20h]
  char v10; // [rsp+11h] [rbp-1Fh]

  if ( a1[77] )
    goto LABEL_2;
  v5 = a1[25];
  if ( !*(_BYTE *)(v5 + 4502) )
  {
    v6 = a1[24];
    v10 = 1;
    v8 = "cu_begin";
    v9 = 3;
    v7 = sub_396F530(v6, (__int64)&v8);
    a1[78] = v7;
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1[24] + 256) + 176LL))(
      *(_QWORD *)(a1[24] + 256),
      v7,
      0);
    if ( a1[77] )
    {
LABEL_2:
      v3 = 5;
      goto LABEL_3;
    }
    v5 = a1[25];
  }
  if ( !*(_BYTE *)(v5 + 4513) )
  {
    sub_39A2D30((__int64)a1, a2, 1u);
    return sub_398C0A0(a1[25]);
  }
  v3 = 4;
LABEL_3:
  sub_39A2D30((__int64)a1, a2, v3);
  result = sub_398C0A0(a1[25]);
  if ( (unsigned __int16)result > 4u )
    return sub_396F360(a1[24], a1[116]);
  return result;
}
