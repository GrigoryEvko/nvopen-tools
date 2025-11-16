// Function: sub_22F5A00
// Address: 0x22f5a00
//
__int64 *__fastcall sub_22F5A00(
        __int64 *a1,
        __int64 a2,
        __int64 (__fastcall ***a3)(_QWORD),
        unsigned int *a4,
        __int64 a5)
{
  char *v6; // rax
  size_t v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned int v10; // r15d
  int v11; // r14d
  int v12; // edx
  int v13; // r13d
  __int64 v14; // rax
  __int64 v15; // r12
  _DWORD *v17; // r15
  __int64 v18; // rbx
  unsigned __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  size_t v22; // rcx
  size_t v23; // rsi
  char *v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  unsigned int *i; // r12
  unsigned int v29; // r14d
  char *v30; // r14
  unsigned int v31; // r12d
  bool v32; // zf
  int v33; // esi
  int v34; // r15d
  int v35; // edx
  int v36; // r13d
  __int64 v37; // rax
  __int64 v38; // rbx
  __int64 v39; // rdx
  int v40; // eax
  unsigned int v41; // [rsp+14h] [rbp-ACh]
  int v44; // [rsp+20h] [rbp-A0h]
  int v46; // [rsp+28h] [rbp-98h]
  __int64 v47; // [rsp+38h] [rbp-88h] BYREF
  char *v48; // [rsp+40h] [rbp-80h] BYREF
  size_t v49; // [rsp+48h] [rbp-78h]
  _QWORD v50[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v51[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v52; // [rsp+70h] [rbp-50h] BYREF
  __int64 v53; // [rsp+78h] [rbp-48h]
  __int64 v54; // [rsp+80h] [rbp-40h]

  v41 = *a4;
  v6 = (char *)(**a3)(a3);
  v7 = 0;
  v48 = v6;
  v8 = (__int64)v6;
  if ( v6 )
    v7 = strlen(v6);
  v9 = *(_QWORD *)(a2 + 80);
  v49 = v7;
  v52 = v9;
  v53 = *(unsigned int *)(a2 + 88);
  if ( (unsigned __int8)sub_22F50A0(&v52, v48, v7) )
  {
    v10 = (*a4)++;
    v11 = sub_22F59B0(a2, *(_DWORD *)(a2 + 64));
    v13 = v12;
    v14 = sub_22077B0(0x58u);
    v15 = v14;
    if ( v14 )
      sub_314D360(v14, v11, v13, (_DWORD)v48, v49, v10, v8, 0);
    *a1 = v15;
  }
  else
  {
    v17 = (_DWORD *)(*(_QWORD *)(a2 + 32) + 80LL * *(unsigned int *)(a2 + 72));
    v18 = *(_QWORD *)(a2 + 32) + 80LL * *(_QWORD *)(a2 + 40);
    v19 = sub_C935B0(&v48, *(unsigned __int8 **)(a2 + 144), *(_QWORD *)(a2 + 152), 0);
    v22 = v49;
    v23 = 0;
    if ( v19 < v49 )
    {
      v23 = v49 - v19;
      v22 = v19;
    }
    v24 = &v48[v22];
    v25 = *(__int64 **)(a2 + 8);
    v26 = *(_QWORD *)(a2 + 16);
    v50[0] = v24;
    v27 = *(_QWORD *)(a2 + 24);
    v53 = v26;
    v54 = v27;
    v50[1] = v23;
    v52 = (__int64)v25;
    for ( i = sub_22F5220(v17, v18, (__int64)v50, v26, v20, v21, v25, v26); (unsigned int *)v18 != i; i += 20 )
    {
      v29 = sub_22F5520(*(void ***)(a2 + 8), *(_QWORD *)(a2 + 16), i, v48, v49, *(_BYTE *)(a2 + 48));
      if ( v29 )
      {
        sub_22F3E40(v51, (__int64)i, a2);
        if ( !*(_QWORD *)(a5 + 16) )
          sub_4263D6(v51, i, v39);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64 *))(a5 + 24))(a5, v51) )
        {
          v40 = ((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(a3, *a4);
          sub_22F4690(&v47, v51, a3, v40, (char *)v29, 0, a4);
          if ( v47 )
          {
            *a1 = v47;
            return a1;
          }
          if ( *a4 != v41 )
          {
            *a1 = 0;
            return a1;
          }
        }
      }
    }
    v30 = v48;
    v31 = *a4;
    v32 = *v48 == 47;
    ++*a4;
    if ( v32 )
      v33 = *(_DWORD *)(a2 + 64);
    else
      v33 = *(_DWORD *)(a2 + 68);
    v34 = sub_22F59B0(a2, v33);
    v36 = v35;
    v44 = (int)v48;
    v46 = v49;
    v37 = sub_22077B0(0x58u);
    v38 = v37;
    if ( v37 )
      sub_314D360(v37, v34, v36, v44, v46, v31, (__int64)v30, 0);
    *a1 = v38;
  }
  return a1;
}
