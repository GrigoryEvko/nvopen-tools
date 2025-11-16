// Function: sub_2950AF0
// Address: 0x2950af0
//
__int64 __fastcall sub_2950AF0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v6; // r8
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // rdx
  _BYTE *v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r14
  const char *v19; // rax
  __int64 v20; // rdx
  __int64 result; // rax
  const char *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // [rsp+0h] [rbp-70h]
  __int64 v25; // [rsp+0h] [rbp-70h]
  __int64 v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+8h] [rbp-68h]
  const void *v28; // [rsp+8h] [rbp-68h]
  const char *v29; // [rsp+10h] [rbp-60h] BYREF
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int16 v31; // [rsp+30h] [rbp-40h]

  v6 = (const void *)(a1 + 96);
  v7 = a2 - 1;
  v8 = 8 * v7;
  v9 = 8LL * a2;
  v10 = *(_QWORD *)a1;
  v11 = *(_BYTE **)(*(_QWORD *)a1 + v9);
  if ( !a2 )
  {
LABEL_11:
    result = sub_29509B0(a1, (__int64)v11);
    *(_QWORD *)(*(_QWORD *)a1 + v9) = result;
    return result;
  }
  for ( ; (unsigned __int8)(*v11 - 67) <= 0xCu; LODWORD(v7) = v7 - 1 )
  {
    v12 = *(unsigned int *)(a1 + 88);
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
    {
      v28 = v6;
      sub_C8D5F0(a1 + 80, v6, v12 + 1, 8u, (__int64)v6, a6);
      v12 = *(unsigned int *)(a1 + 88);
      v6 = v28;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v12) = v11;
    v13 = *(_QWORD *)a1;
    ++*(_DWORD *)(a1 + 88);
    *(_QWORD *)(v13 + v9) = 0;
    v10 = *(_QWORD *)a1;
    v9 = v8;
    v11 = *(_BYTE **)(*(_QWORD *)a1 + v8);
    if ( !(_DWORD)v7 )
      goto LABEL_11;
    v8 -= 8;
  }
  v24 = *((_QWORD *)v11 - 8);
  v26 = *(_QWORD *)(v10 + 8LL * (unsigned int)v7);
  v14 = sub_29509B0(a1, *(_QWORD *)&v11[32 * (v26 == v24) - 64]);
  v15 = sub_2950AF0(a1, (unsigned int)v7);
  v16 = v26;
  v17 = v24;
  v18 = v15;
  v25 = *(_QWORD *)(a1 + 224);
  v27 = *(unsigned __int16 *)(a1 + 232);
  if ( v16 == v17 )
  {
    v22 = sub_BD5D20((__int64)v11);
    v31 = 261;
    v30 = v23;
    v29 = v22;
    result = sub_B504D0((unsigned int)(unsigned __int8)*v11 - 29, v18, v14, (__int64)&v29, v25, v27);
  }
  else
  {
    v19 = sub_BD5D20((__int64)v11);
    v30 = v20;
    v31 = 261;
    v29 = v19;
    result = sub_B504D0((unsigned int)(unsigned __int8)*v11 - 29, v14, v18, (__int64)&v29, v25, v27);
  }
  *(_QWORD *)(*(_QWORD *)a1 + v9) = result;
  return result;
}
