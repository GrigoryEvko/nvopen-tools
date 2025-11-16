// Function: sub_2E46F70
// Address: 0x2e46f70
//
bool __fastcall sub_2E46F70(_QWORD *a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 v7; // rdi
  bool result; // al
  __int64 v9; // rax
  __int64 v10; // r14
  unsigned int v11; // r11d
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int16 *v15; // rax
  __int16 *v16; // rax
  int v17; // edx
  int v18; // r12d
  __int64 v19; // r12
  unsigned int v20; // r15d
  __int64 v21; // rax
  unsigned int v22; // [rsp+10h] [rbp-C0h]
  unsigned int v23; // [rsp+14h] [rbp-BCh]
  _QWORD *v24; // [rsp+18h] [rbp-B8h]
  int v25; // [rsp+2Ch] [rbp-A4h] BYREF
  _QWORD v26[4]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v27; // [rsp+50h] [rbp-80h] BYREF
  __int64 v28; // [rsp+58h] [rbp-78h]
  __int64 v29; // [rsp+70h] [rbp-60h] BYREF
  __int16 *v30; // [rsp+78h] [rbp-58h]
  __int16 v31; // [rsp+80h] [rbp-50h]
  int v32; // [rsp+88h] [rbp-48h]
  __int64 v33; // [rsp+90h] [rbp-40h]
  __int16 v34; // [rsp+98h] [rbp-38h]

  v7 = *(_QWORD *)(a1[2] + 384LL);
  if ( (*(_QWORD *)(v7 + 8LL * (a3 >> 6)) & (1LL << a3)) != 0 )
    return 0;
  if ( (*(_QWORD *)(v7 + 8LL * (a4 >> 6)) & (1LL << a4)) != 0 )
    return 0;
  v9 = sub_2E465E0((__int64)(a1 + 22), a2, a4, *a1, a1[1], *((_BYTE *)a1 + 24));
  v10 = v9;
  if ( !v9 )
    return 0;
  sub_2E44C10((__int64)v26, v9, a1[1], *((_BYTE *)a1 + 24));
  if ( (((*(_BYTE *)(v26[0] + 3LL) & 0x10) != 0) & (*(_BYTE *)(v26[0] + 3LL) >> 6)) != 0 )
    return 0;
  v24 = (_QWORD *)*a1;
  sub_2E44C10((__int64)&v27, v10, a1[1], *((_BYTE *)a1 + 24));
  v11 = *(_DWORD *)(v28 + 8);
  v23 = *(_DWORD *)(v27 + 8);
  if ( a3 == v11 && a4 == *(_DWORD *)(v27 + 8) )
    goto LABEL_13;
  v12 = v24[1];
  v25 = *(_DWORD *)(v28 + 8);
  v22 = v11;
  v13 = *(unsigned int *)(v12 + 24LL * a3 + 8);
  v14 = v24[7];
  v32 = 0;
  v33 = 0;
  v15 = (__int16 *)(v14 + 2 * v13);
  LODWORD(v13) = *v15;
  v16 = v15 + 1;
  LOWORD(v12) = v13;
  v17 = a3 + v13;
  LODWORD(v29) = v17;
  if ( !(_WORD)v12 )
    v16 = 0;
  v31 = v17;
  v34 = 0;
  v30 = v16;
  result = sub_2E46590((int *)&v29, &v25);
  if ( result )
  {
    v18 = sub_E91E30(v24, v22, a3);
    if ( v18 == (unsigned int)sub_E91E30(v24, v23, a4) )
    {
LABEL_13:
      v19 = v10;
      sub_2E44C10((__int64)&v29, a2, a1[1], *((_BYTE *)a1 + 24));
      v20 = *(_DWORD *)(v29 + 8);
      if ( a2 != v10 )
      {
        do
        {
          sub_2E8D6E0(v19, v20, *a1);
          v19 = *(_QWORD *)(v19 + 8);
        }
        while ( a2 != v19 );
      }
      if ( (v30[2] & 1) == 0 )
      {
        v21 = *(_QWORD *)(v10 + 32) + 40LL * (unsigned int)sub_2EAB0A0(v26[1]);
        *(_BYTE *)(v21 + 4) &= ~1u;
      }
      sub_2E88E20(a2);
      *((_BYTE *)a1 + 240) = 1;
      return 1;
    }
    return 0;
  }
  return result;
}
