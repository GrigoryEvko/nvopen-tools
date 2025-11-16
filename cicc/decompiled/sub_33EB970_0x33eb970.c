// Function: sub_33EB970
// Address: 0x33eb970
//
__int64 __fastcall sub_33EB970(__int64 a1, __int64 a2, unsigned int a3)
{
  size_t *v3; // r12
  int v5; // eax
  __int64 v7; // rax
  size_t v8; // rdx
  _QWORD *v9; // rax
  bool v10; // zf
  _QWORD *v11; // rax
  const void *v12; // r13
  size_t v13; // r12
  int v14; // eax
  int v15; // eax
  __int64 v16; // rdx
  size_t **v17; // rax
  size_t v18; // rax
  __int64 v19; // rdi
  const char *v20; // r15
  size_t v21; // rax
  size_t v22; // r8
  _QWORD *v23; // rdx
  int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // rdi
  int v27; // ecx
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // r8
  __int64 v31; // rax
  _QWORD *v32; // rdi
  __int64 v33; // rax
  size_t *v34; // rdx
  size_t *v35; // r13
  int *v36; // rdi
  int *v37; // rax
  __int64 v38; // rax
  int v39; // eax
  int v40; // r9d
  size_t na; // [rsp+8h] [rbp-78h]
  size_t n; // [rsp+8h] [rbp-78h]
  size_t v43; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v44; // [rsp+20h] [rbp-60h] BYREF
  size_t v45; // [rsp+28h] [rbp-58h]
  _QWORD v46[2]; // [rsp+30h] [rbp-50h] BYREF
  int v47; // [rsp+40h] [rbp-40h]

  v5 = *(_DWORD *)(a2 + 24);
  if ( v5 > 44 )
  {
    LODWORD(v3) = 0;
    if ( v5 == 328 )
      return (unsigned int)v3;
  }
  else if ( v5 > 6 )
  {
    switch ( v5 )
    {
      case 7:
        v7 = *(unsigned __int16 *)(a2 + 96);
        v8 = *(_QWORD *)(a2 + 104);
        LOWORD(v44) = v7;
        v45 = v8;
        if ( (_WORD)v7 )
        {
          v9 = (_QWORD *)(*(_QWORD *)(a1 + 848) + 8 * v7);
          v10 = *v9 == 0;
          *v9 = 0;
          LOBYTE(v3) = !v10;
          return (unsigned int)v3;
        }
        v33 = sub_33EB890(a1 + 872, (__int16 *)&v44);
        v35 = v34;
        v3 = (size_t *)v33;
        n = *(_QWORD *)(a1 + 912);
        if ( v33 == *(_QWORD *)(a1 + 896) && v34 == (size_t *)(a1 + 880) )
        {
          sub_33C8980(*(_QWORD *)(a1 + 888));
          *(_QWORD *)(a1 + 888) = 0;
          *(_QWORD *)(a1 + 896) = v35;
          LOBYTE(v3) = n != 0;
          *(_QWORD *)(a1 + 904) = v35;
          *(_QWORD *)(a1 + 912) = 0;
        }
        else
        {
          if ( v34 == (size_t *)v33 )
            goto LABEL_24;
          do
          {
            v36 = (int *)v3;
            v3 = (size_t *)sub_220EF30((__int64)v3);
            v37 = sub_220F330(v36, (_QWORD *)(a1 + 880));
            j_j___libc_free_0((unsigned __int64)v37);
            v38 = *(_QWORD *)(a1 + 912) - 1LL;
            *(_QWORD *)(a1 + 912) = v38;
          }
          while ( v35 != v3 );
          LOBYTE(v3) = n != v38;
        }
        return (unsigned int)v3;
      case 8:
        v11 = (_QWORD *)(*(_QWORD *)(a1 + 824) + 8LL * *(unsigned int *)(a2 + 96));
        v10 = *v11 == 0;
        *v11 = 0;
        LOBYTE(v3) = !v10;
        return (unsigned int)v3;
      case 18:
        v12 = *(const void **)(a2 + 96);
        v13 = 0;
        if ( v12 )
          v13 = strlen(*(const char **)(a2 + 96));
        v14 = sub_C92610();
        v15 = sub_C92860((__int64 *)(a1 + 920), v12, v13, v14);
        if ( v15 == -1 )
          goto LABEL_24;
        v16 = *(_QWORD *)(a1 + 920);
        v17 = (size_t **)(v16 + 8LL * v15);
        if ( v17 == (size_t **)(v16 + 8LL * *(unsigned int *)(a1 + 928)) )
          goto LABEL_24;
        v3 = *v17;
        sub_C929B0(a1 + 920, *v17);
        v18 = *v3;
        v19 = (__int64)v3;
        LODWORD(v3) = 1;
        sub_C7D6A0(v19, v18 + 17, 8);
        return (unsigned int)v3;
      case 42:
        v20 = *(const char **)(a2 + 96);
        LODWORD(v3) = *(_DWORD *)(a2 + 104);
        v44 = v46;
        if ( !v20 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v21 = strlen(v20);
        v43 = v21;
        v22 = v21;
        if ( v21 > 0xF )
        {
          na = v21;
          v31 = sub_22409D0((__int64)&v44, &v43, 0);
          v22 = na;
          v44 = (_QWORD *)v31;
          v32 = (_QWORD *)v31;
          v46[0] = v43;
        }
        else
        {
          if ( v21 == 1 )
          {
            LOBYTE(v46[0]) = *v20;
            v23 = v46;
            goto LABEL_19;
          }
          if ( !v21 )
          {
            v23 = v46;
            goto LABEL_19;
          }
          v32 = v46;
        }
        memcpy(v32, v20, v22);
        v21 = v43;
        v23 = v44;
LABEL_19:
        v45 = v21;
        *((_BYTE *)v23 + v21) = 0;
        v47 = (int)v3;
        LOBYTE(v3) = sub_33EB790(a1 + 944, (__int64)&v44) != 0;
        if ( v44 != v46 )
          j_j___libc_free_0((unsigned __int64)v44);
        return (unsigned int)v3;
      case 44:
        v24 = *(_DWORD *)(a1 + 1016);
        v25 = *(_QWORD *)(a2 + 96);
        v26 = *(_QWORD *)(a1 + 1000);
        if ( !v24 )
          goto LABEL_24;
        v27 = v24 - 1;
        v28 = (v24 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v29 = (__int64 *)(v26 + 16LL * v28);
        v30 = *v29;
        if ( v25 == *v29 )
          goto LABEL_23;
        v39 = 1;
        break;
      default:
        return sub_C65A50(a1 + 520, (_QWORD *)a2, a3);
    }
    while ( v30 != -4096 )
    {
      v40 = v39 + 1;
      v28 = v27 & (v39 + v28);
      v29 = (__int64 *)(v26 + 16LL * v28);
      v30 = *v29;
      if ( v25 == *v29 )
      {
LABEL_23:
        *v29 = -8192;
        LODWORD(v3) = 1;
        --*(_DWORD *)(a1 + 1008);
        ++*(_DWORD *)(a1 + 1012);
        return (unsigned int)v3;
      }
      v39 = v40;
    }
LABEL_24:
    LODWORD(v3) = 0;
    return (unsigned int)v3;
  }
  return sub_C65A50(a1 + 520, (_QWORD *)a2, a3);
}
