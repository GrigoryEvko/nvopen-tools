// Function: sub_285D500
// Address: 0x285d500
//
_QWORD *__fastcall sub_285D500(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, int a6)
{
  __int16 v7; // ax
  _QWORD *v11; // r10
  unsigned int v12; // ebx
  _QWORD *v13; // r15
  __int64 *v14; // rax
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 *v18; // rax
  __int64 *v19; // r10
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 *v31; // rbx
  __int64 v32; // rcx
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 *v35; // rax
  __int64 v36; // rax
  __int64 *v37; // [rsp+8h] [rbp-68h]
  __int64 *v38; // [rsp+8h] [rbp-68h]
  _QWORD *v39; // [rsp+10h] [rbp-60h]
  __int64 v41; // [rsp+18h] [rbp-58h]
  __int64 v42; // [rsp+18h] [rbp-58h]
  unsigned __int64 v43[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v44[8]; // [rsp+30h] [rbp-40h] BYREF

  if ( a6 == 3 )
    return (_QWORD *)a1;
  v7 = *(_WORD *)(a1 + 24);
  if ( v7 == 5 )
  {
    v11 = *(_QWORD **)(a1 + 32);
    v12 = a6 + 1;
    v13 = v11;
    v39 = &v11[*(_QWORD *)(a1 + 40)];
    if ( v39 != v11 )
    {
      do
      {
        v14 = (__int64 *)sub_285D500(*v13, a2, a3, a4, a5, v12);
        if ( v14 )
        {
          if ( a2 )
          {
            v44[1] = v14;
            v43[0] = (unsigned __int64)v44;
            v44[0] = a2;
            v43[1] = 0x200000002LL;
            v14 = sub_DC8BD0(a5, (__int64)v43, 0, 0);
            if ( (_QWORD *)v43[0] != v44 )
            {
              v37 = v14;
              _libc_free(v43[0]);
              v14 = v37;
            }
          }
          v16 = *(unsigned int *)(a3 + 8);
          if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            v38 = v14;
            sub_C8D5F0(a3, (const void *)(a3 + 16), v16 + 1, 8u, v16 + 1, v15);
            v16 = *(unsigned int *)(a3 + 8);
            v14 = v38;
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v16) = v14;
          ++*(_DWORD *)(a3 + 8);
        }
        ++v13;
      }
      while ( v39 != v13 );
    }
    return 0;
  }
  if ( v7 == 8 )
  {
    if ( sub_D968A0(**(_QWORD **)(a1 + 32)) || *(_QWORD *)(a1 + 40) != 2 )
      return (_QWORD *)a1;
    v27 = sub_285D500(**(_QWORD **)(a1 + 32), a2, a3, a4, a5, (unsigned int)(a6 + 1));
    v31 = (__int64 *)v27;
    if ( v27 )
    {
      v32 = *(_QWORD *)(a1 + 48);
      if ( a4 == v32 || *(_WORD *)(v27 + 24) != 8 )
      {
        if ( a2 )
          v31 = sub_DCA690(a5, a2, v27, 0, 0);
        sub_D9B3A0(a3, (__int64)v31, v28, v32, v29, v30);
        v33 = **(_QWORD **)(a1 + 32);
        if ( !v33 )
          return (_QWORD *)a1;
        goto LABEL_30;
      }
      if ( v27 != **(_QWORD **)(a1 + 32) )
        goto LABEL_31;
    }
    else
    {
      v33 = **(_QWORD **)(a1 + 32);
      if ( v33 )
      {
LABEL_30:
        v34 = sub_D95540(v33);
        v35 = sub_DA2C50((__int64)a5, v34, 0, 0);
        v32 = *(_QWORD *)(a1 + 48);
        v31 = v35;
LABEL_31:
        v42 = v32;
        v36 = sub_D33D80((_QWORD *)a1, (__int64)a5, v28, v32, v29);
        return sub_DC1960((__int64)a5, (__int64)v31, v36, v42, 0);
      }
    }
    return (_QWORD *)a1;
  }
  if ( v7 != 6 )
    return (_QWORD *)a1;
  if ( *(_QWORD *)(a1 + 40) != 2 )
    return (_QWORD *)a1;
  v18 = *(__int64 **)(a1 + 32);
  v19 = (__int64 *)*v18;
  if ( *(_WORD *)(*v18 + 24) )
    return (_QWORD *)a1;
  if ( a2 )
  {
    v19 = sub_DCA690(a5, a2, *v18, 0, 0);
    v18 = *(__int64 **)(a1 + 32);
  }
  v20 = a4;
  v41 = (__int64)v19;
  v21 = sub_285D500(v18[1], v19, a3, v20, a5, (unsigned int)(a6 + 1));
  if ( !v21 )
    return 0;
  v22 = sub_DCA690(a5, v41, v21, 0, 0);
  sub_D9B3A0(a3, (__int64)v22, v23, v24, v25, v26);
  return 0;
}
