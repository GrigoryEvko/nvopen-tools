// Function: sub_EDDA30
// Address: 0xedda30
//
__int64 *__fastcall sub_EDDA30(__int64 *a1, __int64 a2, __int64 *a3)
{
  bool (__fastcall *v4)(__int64); // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  int v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = *(bool (__fastcall **)(__int64))(*(_QWORD *)a2 + 40LL);
  if ( v4 == sub_ED6000 )
  {
    if ( !*(_QWORD *)(a2 + 32) )
    {
LABEL_3:
      v8[0] = 1;
      sub_ED8A30(a1, v8);
      return a1;
    }
  }
  else if ( v4(a2) )
  {
    goto LABEL_3;
  }
  v6 = *(_QWORD *)(a2 + 16) + (*(_QWORD *)(a2 + 24) == 0 ? 10LL : 8LL);
  *a3 = sub_EDD200(
          *(_QWORD **)(a2 + 40),
          v6 + 16,
          *(_QWORD *)v6,
          (unsigned int *)(v6 + 16 + *(_QWORD *)v6),
          *(_QWORD *)(v6 + 8));
  a3[1] = v7;
  if ( v7 )
  {
    *a1 = 1;
  }
  else
  {
    v8[0] = 9;
    sub_ED89C0(a1, v8, "profile data is empty");
  }
  return a1;
}
