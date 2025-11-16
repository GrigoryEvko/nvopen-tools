// Function: sub_257ADC0
// Address: 0x257adc0
//
char __fastcall sub_257ADC0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, __int64 a5)
{
  char v5; // al
  __int64 v6; // r14
  __int64 **v7; // rax
  __int64 **v8; // rax
  __int64 *v10; // rax
  __int64 *v11; // [rsp+18h] [rbp-78h] BYREF
  __int64 v12[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v13; // [rsp+30h] [rbp-60h]
  __int64 v14; // [rsp+38h] [rbp-58h]
  __int64 *v15[10]; // [rsp+40h] [rbp-50h] BYREF

  v12[0] = (__int64)a3;
  v5 = *(_BYTE *)(a1 + 97);
  v12[1] = (__int64)a4;
  v13 = a5;
  v14 = 0;
  if ( !a5 || *(_DWORD *)(a5 + 20) == *(_DWORD *)(a5 + 24) )
  {
    v13 = 0;
    v6 = a1 + 168;
    if ( v5 )
      goto LABEL_7;
    return 1;
  }
  if ( !v5 )
    return 1;
  v6 = a1 + 168;
  v15[0] = a3;
  v15[1] = a4;
  v15[2] = 0;
  v15[3] = 0;
  v11 = (__int64 *)v15;
  v7 = sub_2568130(a1 + 168, (__int64 *)&v11);
  if ( v7 && v7 != (__int64 **)(*(_QWORD *)(a1 + 176) + 8LL * *(unsigned int *)(a1 + 192)) && !*((_DWORD *)*v7 + 6) )
    return 0;
LABEL_7:
  v15[0] = v12;
  v8 = sub_2568130(v6, (__int64 *)v15);
  if ( v8 && v8 != (__int64 **)(*(_QWORD *)(a1 + 176) + 8LL * *(unsigned int *)(a1 + 192)) )
    return *((_DWORD *)*v8 + 6) == 1;
  v11 = v12;
  if ( !sub_25682F0(v6, (__int64 *)&v11, v15) )
  {
    v10 = sub_2576D50(v6, (__int64 *)&v11, v15[0]);
    *v10 = (__int64)v11;
  }
  return sub_257AC00(a1, a2, v12, 1u);
}
