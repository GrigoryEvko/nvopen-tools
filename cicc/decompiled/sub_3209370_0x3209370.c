// Function: sub_3209370
// Address: 0x3209370
//
__int64 __fastcall sub_3209370(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  __int64 *v4; // r15
  __int64 v5; // r12
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 *v10; // rsi
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // r14
  void (*v15)(); // r12
  const char *v16; // rax
  __int64 v17; // rdx
  const char *v18; // rsi
  __int64 v19; // rdi
  const char *v20; // rcx
  void (*v21)(); // rax
  __int64 v22; // rax
  __int64 *v23; // r12
  __int64 v24; // r14
  __int64 *v25; // r13
  __int64 *v26; // rsi
  __int64 *i; // [rsp+10h] [rbp-70h]
  unsigned __int64 v28; // [rsp+18h] [rbp-68h]
  _QWORD v29[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v30; // [rsp+40h] [rbp-40h]

  sub_3200CF0(a1, 0);
  if ( *(_DWORD *)(a1 + 944) || (v2 = *(unsigned int *)(a1 + 976), (_DWORD)v2) )
  {
    v19 = *(_QWORD *)(a1 + 528);
    v20 = "Symbol subsection for globals";
    v21 = *(void (**)())(*(_QWORD *)v19 + 120LL);
    v29[0] = "Symbol subsection for globals";
    v30 = 259;
    if ( v21 != nullsub_98 )
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v21)(v19, v29, 1);
    v22 = sub_31F8650(a1, 241, v2, (__int64)v20, v3);
    v23 = *(__int64 **)(a1 + 936);
    v24 = v22;
    v25 = &v23[2 * *(unsigned int *)(a1 + 944)];
    while ( v25 != v23 )
    {
      v26 = v23;
      v23 += 2;
      sub_3208CF0(a1, v26);
    }
    sub_32089B0(a1);
    sub_31F8740(a1, v24);
  }
  v4 = *(__int64 **)(a1 + 904);
  v5 = 2LL * *(unsigned int *)(a1 + 912);
  result = (__int64)&v4[v5];
  for ( i = &v4[v5]; v4 != i; result = sub_31F8740(a1, v11) )
  {
    v28 = v4[1] & 0xFFFFFFFFFFFFFFF8LL;
    v12 = sub_31DB510(*(_QWORD *)(a1 + 8), v28);
    v13 = *(_QWORD *)(a1 + 528);
    v14 = v12;
    v15 = *(void (**)())(*(_QWORD *)v13 + 120LL);
    v16 = sub_BD5D20(v28);
    v18 = v16;
    if ( v17 && *v16 == 1 )
    {
      --v17;
      v18 = v16 + 1;
    }
    v29[2] = v18;
    v29[0] = "Symbol subsection for ";
    v29[3] = v17;
    v30 = 1283;
    if ( v15 != nullsub_98 )
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v15)(v13, v29, 1);
    sub_3200CF0(a1, v14);
    v10 = v4;
    v4 += 2;
    v11 = sub_31F8650(a1, 241, v7, v8, v9);
    sub_3208CF0(a1, v10);
  }
  return result;
}
