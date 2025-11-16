// Function: sub_2411210
// Address: 0x2411210
//
unsigned __int8 *__fastcall sub_2411210(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // al
  unsigned __int8 *v5; // r14
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int8 *v12; // r13
  __int64 (__fastcall *v13)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned int *v14; // r12
  __int64 v15; // r13
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned __int8 *v23; // r13
  __int64 (__fastcall *v24)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned int *v25; // r12
  __int64 v26; // r13
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v32; // [rsp+28h] [rbp-A8h]
  unsigned int v33; // [rsp+3Ch] [rbp-94h] BYREF
  _DWORD v34[8]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v35; // [rsp+60h] [rbp-70h]
  _BYTE v36[32]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v37; // [rsp+90h] [rbp-40h]

  v32 = *(_QWORD *)(a2 + 8);
  v4 = *(_BYTE *)(v32 + 8);
  if ( v4 == 16 )
  {
    if ( !*(_QWORD *)(v32 + 32) )
      return *(unsigned __int8 **)(*(_QWORD *)a1 + 72LL);
    v34[0] = 0;
    v37 = 257;
    v7 = sub_94D3D0((unsigned int **)a3, a2, (__int64)v34, 1, (__int64)v36);
    v33 = 1;
    v5 = (unsigned __int8 *)sub_2411210(a1, v7, a3);
    if ( *(_QWORD *)(v32 + 32) <= 1u )
      return v5;
    while ( 1 )
    {
      v37 = 257;
      v9 = sub_94D3D0((unsigned int **)a3, a2, (__int64)&v33, 1, (__int64)v36);
      v10 = sub_2411210(a1, v9, a3);
      v11 = *(_QWORD *)(a3 + 80);
      v35 = 257;
      v12 = (unsigned __int8 *)v10;
      v13 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v11 + 16LL);
      if ( v13 == sub_9202E0 )
      {
        if ( *v5 > 0x15u || *v12 > 0x15u )
        {
LABEL_17:
          v37 = 257;
          v5 = (unsigned __int8 *)sub_B504D0(29, (__int64)v5, (__int64)v12, (__int64)v36, 0, 0);
          (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
            *(_QWORD *)(a3 + 88),
            v5,
            v34,
            *(_QWORD *)(a3 + 56),
            *(_QWORD *)(a3 + 64));
          v14 = *(unsigned int **)a3;
          v15 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
          if ( *(_QWORD *)a3 != v15 )
          {
            do
            {
              v16 = *((_QWORD *)v14 + 1);
              v17 = *v14;
              v14 += 4;
              sub_B99FD0((__int64)v5, v17, v16);
            }
            while ( (unsigned int *)v15 != v14 );
          }
          goto LABEL_14;
        }
        if ( (unsigned __int8)sub_AC47B0(29) )
          v8 = sub_AD5570(29, (__int64)v5, v12, 0, 0);
        else
          v8 = sub_AABE40(0x1Du, v5, v12);
      }
      else
      {
        v8 = v13(v11, 29u, v5, v12);
      }
      if ( !v8 )
        goto LABEL_17;
      v5 = (unsigned __int8 *)v8;
LABEL_14:
      if ( (unsigned __int64)++v33 >= *(_QWORD *)(v32 + 32) )
        return v5;
    }
  }
  v5 = (unsigned __int8 *)a2;
  if ( v4 == 15 )
  {
    if ( !*(_DWORD *)(v32 + 12) )
      return *(unsigned __int8 **)(*(_QWORD *)a1 + 72LL);
    v37 = 257;
    v34[0] = 0;
    v18 = sub_94D3D0((unsigned int **)a3, a2, (__int64)v34, 1, (__int64)v36);
    v33 = 1;
    v5 = (unsigned __int8 *)sub_2411210(a1, v18, a3);
    if ( *(_DWORD *)(v32 + 12) <= 1u )
      return v5;
    do
    {
      v37 = 257;
      v20 = sub_94D3D0((unsigned int **)a3, a2, (__int64)&v33, 1, (__int64)v36);
      v21 = sub_2411210(a1, v20, a3);
      v22 = *(_QWORD *)(a3 + 80);
      v35 = 257;
      v23 = (unsigned __int8 *)v21;
      v24 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v22 + 16LL);
      if ( v24 == sub_9202E0 )
      {
        if ( *v5 > 0x15u || *v23 > 0x15u )
        {
LABEL_30:
          v37 = 257;
          v5 = (unsigned __int8 *)sub_B504D0(29, (__int64)v5, (__int64)v23, (__int64)v36, 0, 0);
          (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
            *(_QWORD *)(a3 + 88),
            v5,
            v34,
            *(_QWORD *)(a3 + 56),
            *(_QWORD *)(a3 + 64));
          v25 = *(unsigned int **)a3;
          v26 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
          if ( *(_QWORD *)a3 != v26 )
          {
            do
            {
              v27 = *((_QWORD *)v25 + 1);
              v28 = *v25;
              v25 += 4;
              sub_B99FD0((__int64)v5, v28, v27);
            }
            while ( (unsigned int *)v26 != v25 );
          }
          goto LABEL_27;
        }
        if ( (unsigned __int8)sub_AC47B0(29) )
          v19 = sub_AD5570(29, (__int64)v5, v23, 0, 0);
        else
          v19 = sub_AABE40(0x1Du, v5, v23);
      }
      else
      {
        v19 = v24(v22, 29u, v5, v23);
      }
      if ( !v19 )
        goto LABEL_30;
      v5 = (unsigned __int8 *)v19;
LABEL_27:
      ++v33;
    }
    while ( *(_DWORD *)(v32 + 12) > v33 );
  }
  return v5;
}
