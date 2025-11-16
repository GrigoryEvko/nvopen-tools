// Function: sub_3264B60
// Address: 0x3264b60
//
__int64 __fastcall sub_3264B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v7; // eax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // r14d
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // eax
  __int64 v19; // rax
  unsigned int v20; // esi
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // eax
  __int64 (__fastcall *v24)(__int64, __int64, __int64 *, __int64, _QWORD, _QWORD, int, __int64); // r15
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // eax
  int v29; // [rsp+0h] [rbp-70h] BYREF
  __int64 v30; // [rsp+8h] [rbp-68h]
  __int64 v31; // [rsp+10h] [rbp-60h] BYREF
  __int64 v32; // [rsp+18h] [rbp-58h]
  char v33; // [rsp+20h] [rbp-50h]
  __int64 v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+30h] [rbp-40h]

  v7 = *(_DWORD *)(a2 + 24);
  v30 = 0;
  if ( v7 == 298 )
  {
    if ( (*(_WORD *)(a2 + 32) & 0x380) != 0 || a1 != *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) )
      return 0;
  }
  else
  {
    if ( v7 != 299 )
    {
      if ( v7 == 362 )
      {
        if ( (*(_WORD *)(a2 + 32) & 0x380) != 0 || *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) != a1 )
          return 0;
        goto LABEL_8;
      }
      if ( v7 != 363 )
        return 0;
    }
    if ( (*(_WORD *)(a2 + 32) & 0x380) != 0 || a1 != *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) )
      return 0;
  }
LABEL_8:
  v9 = *(_QWORD *)(a2 + 104);
  v10 = *(_QWORD *)(a2 + 112);
  LOWORD(v29) = *(_WORD *)(a2 + 96);
  v30 = v9;
  v31 = 0;
  v14 = sub_2EAC1E0(v10);
  v15 = *(_DWORD *)(a1 + 24);
  v32 = 0;
  v34 = 0;
  v35 = 0;
  if ( v15 == 56 )
  {
    v22 = *(_QWORD *)(a1 + 40);
    v33 = 1;
    v17 = *(_QWORD *)(v22 + 40);
    v23 = *(_DWORD *)(v17 + 24);
    if ( v23 == 35 || v23 == 11 )
    {
      v27 = *(_QWORD *)(v17 + 96);
      v17 = *(_QWORD *)(v27 + 24);
      v28 = *(_DWORD *)(v27 + 32);
      if ( v28 > 0x40 )
      {
        v11 = *(_QWORD *)v17;
      }
      else
      {
        v11 = 0;
        if ( v28 )
        {
          v17 = v17 << (64 - (unsigned __int8)v28) >> (64 - (unsigned __int8)v28);
          v11 = v17;
        }
      }
      v32 = v11;
      goto LABEL_25;
    }
    goto LABEL_24;
  }
  if ( v15 != 57 )
    return 0;
  v16 = *(_QWORD *)(a1 + 40);
  v33 = 1;
  v17 = *(_QWORD *)(v16 + 40);
  v18 = *(_DWORD *)(v17 + 24);
  if ( v18 == 11 || v18 == 35 )
  {
    v19 = *(_QWORD *)(v17 + 96);
    v20 = *(_DWORD *)(v19 + 32);
    v17 = *(_QWORD *)(v19 + 24);
    if ( v20 > 0x40 )
    {
      v21 = -*(_QWORD *)v17;
    }
    else
    {
      v21 = 0;
      if ( v20 )
      {
        v11 = 64 - v20;
        v17 = v17 << (64 - (unsigned __int8)v20) >> (64 - (unsigned __int8)v20);
        v21 = -v17;
      }
    }
    v32 = v21;
    goto LABEL_25;
  }
LABEL_24:
  v34 = 1;
LABEL_25:
  v24 = *(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64, _QWORD, _QWORD, int, __int64))(*(_QWORD *)a4 + 1288LL);
  v25 = sub_3007410((__int64)&v29, *(__int64 **)(a3 + 64), v17, v11, v12, v13);
  v26 = sub_2E79000(*(__int64 **)(a3 + 40));
  return v24(a4, v26, &v31, v25, v14, 0, v29, v30);
}
