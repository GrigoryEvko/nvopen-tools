// Function: sub_240F6F0
// Address: 0x240f6f0
//
__int64 __fastcall sub_240F6F0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  __int64 v5; // r9
  char v9; // al
  __int64 v10; // r12
  __int64 v11; // rax
  unsigned int v12; // r14d
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v16; // rdi
  _DWORD *v17; // r10
  __int64 v18; // rbx
  __int64 (__fastcall *v19)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rax
  unsigned __int64 v27; // r14
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-98h]
  _DWORD *v32; // [rsp+8h] [rbp-98h]
  const void *v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+8h] [rbp-98h]
  __int64 v35; // [rsp+8h] [rbp-98h]
  __int64 v36; // [rsp+8h] [rbp-98h]
  _DWORD *v37; // [rsp+8h] [rbp-98h]
  _BYTE v38[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v39; // [rsp+30h] [rbp-70h]
  _BYTE v40[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v41; // [rsp+60h] [rbp-40h]

  v5 = a3;
  v9 = *(_BYTE *)(a3 + 8);
  if ( v9 != 16 )
  {
    if ( v9 == 15 )
    {
      v10 = a1;
      if ( *(_DWORD *)(a3 + 12) )
      {
        v11 = *(unsigned int *)(a2 + 8);
        v12 = 0;
        do
        {
          if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
          {
            v36 = v5;
            sub_C8D5F0(a2, (const void *)(a2 + 16), v11 + 1, 4u, a5, v5);
            v11 = *(unsigned int *)(a2 + 8);
            v5 = v36;
          }
          v31 = v5;
          *(_DWORD *)(*(_QWORD *)a2 + 4 * v11) = v12;
          v13 = v12++;
          ++*(_DWORD *)(a2 + 8);
          v14 = sub_240F6F0(v10, a2, *(_QWORD *)(*(_QWORD *)(v5 + 16) + 8 * v13), a4, a5);
          v5 = v31;
          v10 = v14;
          v11 = (unsigned int)(*(_DWORD *)(a2 + 8) - 1);
          *(_DWORD *)(a2 + 8) = v11;
        }
        while ( v12 < *(_DWORD *)(v31 + 12) );
      }
      return v10;
    }
    v16 = *(_QWORD *)(a5 + 80);
    v17 = *(_DWORD **)a2;
    v39 = 257;
    v18 = *(unsigned int *)(a2 + 8);
    v19 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v16 + 88LL);
    if ( v19 == sub_9482E0 )
    {
      if ( *(_BYTE *)a1 > 0x15u || *a4 > 0x15u )
      {
LABEL_14:
        v33 = v17;
        v41 = 257;
        v21 = sub_BD2C40(104, unk_3F148BC);
        v10 = (__int64)v21;
        if ( v21 )
        {
          sub_B44260((__int64)v21, *(_QWORD *)(a1 + 8), 65, 2u, 0, 0);
          *(_QWORD *)(v10 + 72) = v10 + 88;
          *(_QWORD *)(v10 + 80) = 0x400000000LL;
          sub_B4FD20(v10, a1, (__int64)a4, v33, v18, (__int64)v40);
        }
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
          *(_QWORD *)(a5 + 88),
          v10,
          v38,
          *(_QWORD *)(a5 + 56),
          *(_QWORD *)(a5 + 64));
        v22 = *(_QWORD *)a5;
        v23 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
        while ( v23 != v22 )
        {
          v24 = *(_QWORD *)(v22 + 8);
          v25 = *(_DWORD *)v22;
          v22 += 16;
          sub_B99FD0(v10, v25, v24);
        }
        return v10;
      }
      v32 = v17;
      v20 = sub_AAAE30(a1, (__int64)a4, v17, *(unsigned int *)(a2 + 8));
      v17 = v32;
      v10 = v20;
    }
    else
    {
      v37 = v17;
      v30 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64, __int64))v19)(
              v16,
              a1,
              a4,
              v17,
              v18,
              a3);
      v17 = v37;
      v10 = v30;
    }
    if ( v10 )
      return v10;
    goto LABEL_14;
  }
  v10 = a1;
  if ( *(_QWORD *)(a3 + 32) )
  {
    v26 = *(unsigned int *)(a2 + 8);
    LODWORD(v27) = 0;
    v28 = v26 + 1;
    if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      goto LABEL_23;
    while ( 1 )
    {
      v34 = v5;
      *(_DWORD *)(*(_QWORD *)a2 + 4 * v26) = v27;
      ++*(_DWORD *)(a2 + 8);
      v29 = sub_240F6F0(v10, a2, *(_QWORD *)(v5 + 24), a4, a5);
      v5 = v34;
      v10 = v29;
      v27 = (unsigned int)(v27 + 1);
      v26 = (unsigned int)(*(_DWORD *)(a2 + 8) - 1);
      *(_DWORD *)(a2 + 8) = v26;
      if ( v27 >= *(_QWORD *)(v34 + 32) )
        break;
      v28 = v26 + 1;
      if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
LABEL_23:
        v35 = v5;
        sub_C8D5F0(a2, (const void *)(a2 + 16), v28, 4u, a5, v5);
        v26 = *(unsigned int *)(a2 + 8);
        v5 = v35;
      }
    }
  }
  return v10;
}
