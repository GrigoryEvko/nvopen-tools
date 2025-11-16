// Function: sub_30F4530
// Address: 0x30f4530
//
__int64 __fastcall sub_30F4530(__int64 a1, char *a2, unsigned int a3)
{
  __int64 result; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  __int64 v8; // r8
  int v9; // eax
  unsigned __int64 v10; // rdx
  _QWORD **v11; // r15
  int v12; // eax
  int v13; // eax
  __int64 *v14; // rax
  __int64 v15; // r13
  __int64 *v16; // r15
  __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r13
  int v24; // ecx
  int v25; // eax
  __int64 v26; // r15
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // r13
  _QWORD *v30; // r14
  _QWORD *v31; // rax
  __int64 v32; // rdi
  _QWORD *v33; // rax
  __int64 *v34; // rdi
  __int64 *v35; // r12
  __int64 v36; // r12
  unsigned int v37; // ebx
  unsigned int v38; // ebx
  _QWORD *v39; // [rsp+0h] [rbp-70h]
  unsigned int v40; // [rsp+Ch] [rbp-64h]
  _QWORD *v41; // [rsp+10h] [rbp-60h] BYREF
  __int64 v42; // [rsp+18h] [rbp-58h]
  _QWORD *v43; // [rsp+20h] [rbp-50h] BYREF
  __int64 v44; // [rsp+28h] [rbp-48h]
  _QWORD *v45; // [rsp+30h] [rbp-40h] BYREF
  _QWORD *v46; // [rsp+38h] [rbp-38h]

  if ( sub_30F4340(a1, (__int64)a2) )
    return 1;
  v6 = sub_30F3560(
         a2,
         *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * *(unsigned int *)(a1 + 72) - 8),
         *(__int64 **)(a1 + 104));
  v41 = 0;
  v7 = v6;
  if ( (unsigned __int8)sub_30F4190(a1, (__int64)a2, (__int64 *)&v41, a3, v8) )
  {
    v26 = *(_QWORD *)(a1 + 104);
    v27 = sub_D95540((__int64)v7);
    v28 = sub_D95540((__int64)v41);
    v29 = sub_D970B0(v26, v28, v27);
    v30 = sub_DA2C50(*(_QWORD *)(a1 + 104), v29, a3, 0);
    v31 = sub_DD2CB0(*(_QWORD *)(a1 + 104), (__int64)v41, v29);
    v32 = *(_QWORD *)(a1 + 104);
    v41 = v31;
    v33 = sub_DC2CB0(v32, (__int64)v7, v29);
    v34 = *(__int64 **)(a1 + 104);
    v46 = v33;
    v45 = v41;
    v43 = &v45;
    v44 = 0x200000002LL;
    v35 = sub_DC8BD0(v34, (__int64)&v43, 0, 0);
    if ( v43 != &v45 )
      _libc_free((unsigned __int64)v43);
    v7 = sub_DD21F0(*(__int64 **)(a1 + 104), (__int64)v35, (__int64)v30);
  }
  else
  {
    v9 = sub_30F4110(a1, (__int64)a2);
    v10 = (unsigned int)(v9 + 1);
    v40 = v9 + 1;
    if ( (unsigned __int64)*(unsigned int *)(a1 + 32) - 1 > v10 )
    {
      do
      {
        v17 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v10);
        if ( *(_WORD *)(v17 + 24) != 8 )
          BUG();
        v18 = sub_30F3560(
                *(char **)(v17 + 48),
                *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * *(unsigned int *)(a1 + 72) - 8),
                *(__int64 **)(a1 + 104));
        v19 = *(_QWORD *)(a1 + 104);
        v20 = (__int64)v18;
        v21 = sub_D95540((__int64)v18);
        v22 = sub_D95540((__int64)v7);
        v23 = sub_D970B0(v19, v22, v21);
        v24 = *(unsigned __int8 *)(v23 + 8);
        if ( (unsigned int)(v24 - 17) <= 1 )
        {
          v11 = *(_QWORD ***)(v23 + 24);
          v12 = *(_DWORD *)(v23 + 32);
          BYTE4(v42) = (_BYTE)v24 == 18;
          LODWORD(v42) = v12;
          v13 = sub_BCB060((__int64)v11);
          v14 = (__int64 *)sub_BCD140(*v11, 2 * v13);
          v15 = sub_BCE1B0(v14, v42);
        }
        else
        {
          v25 = sub_BCB060(v23);
          v15 = sub_BCD140(*(_QWORD **)v23, 2 * v25);
        }
        v16 = *(__int64 **)(a1 + 104);
        v39 = sub_DC2CB0((__int64)v16, v20, v15);
        v45 = sub_DC2CB0(*(_QWORD *)(a1 + 104), (__int64)v7, v15);
        v46 = v39;
        v43 = &v45;
        v44 = 0x200000002LL;
        v7 = sub_DC8BD0(v16, (__int64)&v43, 0, 0);
        if ( v43 != &v45 )
          _libc_free((unsigned __int64)v43);
        v10 = ++v40;
      }
      while ( v40 < (unsigned __int64)*(unsigned int *)(a1 + 32) - 1 );
    }
  }
  if ( *((_WORD *)v7 + 12) )
    return 0;
  v36 = v7[4];
  v37 = *(_DWORD *)(v36 + 32);
  if ( v37 > 0x40 )
  {
    v38 = v37 - sub_C444A0(v36 + 24);
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v38 <= 0x40 )
    {
      result = **(_QWORD **)(v36 + 24);
      if ( result < 0 )
        return 0x7FFFFFFFFFFFFFFFLL;
    }
  }
  else
  {
    result = *(_QWORD *)(v36 + 24);
    if ( result < 0 )
      return 0x7FFFFFFFFFFFFFFFLL;
  }
  return result;
}
