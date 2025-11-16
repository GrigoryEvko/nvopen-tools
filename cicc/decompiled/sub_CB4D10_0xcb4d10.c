// Function: sub_CB4D10
// Address: 0xcb4d10
//
__int64 __fastcall sub_CB4D10(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 **v10; // rax
  __int64 *v11; // r12
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 *v14; // r15
  __int64 *v15; // r14
  __int64 v16; // rdi
  unsigned int v17; // ecx
  __int64 *v18; // r14
  __int64 *v19; // r15
  __int64 v20; // rdi
  __int64 *v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 **v25; // r15
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 *v28; // r14
  __int64 v29; // r14
  __int64 v30; // rdi
  __int64 *v31; // r15
  __int64 *v32; // r14
  __int64 i; // rax
  __int64 v34; // rdi
  unsigned int v35; // ecx
  __int64 *v36; // r14
  __int64 v37; // rdi
  _QWORD *v38; // rax
  unsigned __int64 v39; // [rsp+8h] [rbp-38h]

  while ( 1 )
  {
    v3 = (_QWORD *)sub_CA8A20(*(_QWORD *)(a1 + 80));
    v6 = *(_QWORD *)(a1 + 592);
    if ( v6 && (v7 = *(_QWORD *)v6) != 0 )
    {
      if ( v3 && *v3 && (_QWORD *)v6 == v3 )
        return 0;
      v8 = *(_QWORD *)(v7 + 104);
      if ( !v8 )
        goto LABEL_27;
    }
    else
    {
      if ( !v3 || !*v3 )
        return 0;
      v7 = *(_QWORD *)v6;
      v8 = *(_QWORD *)(*(_QWORD *)v6 + 104LL);
      if ( !v8 )
      {
LABEL_27:
        v8 = sub_CAD820(v7, a2, v6, v4, v5);
        *(_QWORD *)(v7 + 104) = v8;
        if ( !v8 )
        {
          v23 = sub_2241E50(v7, a2, v22, v4, v5);
          *(_DWORD *)(a1 + 96) = 22;
          *(_QWORD *)(a1 + 104) = v23;
          return 0;
        }
      }
    }
    v9 = *(unsigned int *)(v8 + 32);
    if ( (_DWORD)v9 )
      break;
    if ( (unsigned __int8)sub_CAF190(**(__int64 ****)(a1 + 592), a2, v9, v4, v5) )
    {
      v25 = *(__int64 ***)(a1 + 592);
      v26 = **v25;
      v27 = sub_22077B0(160);
      v28 = (__int64 *)v27;
      if ( v27 )
      {
        a2 = v26;
        sub_CAFBE0(v27, v26);
      }
      v11 = *v25;
      *v25 = v28;
      if ( v11 )
      {
        v29 = v11[16];
        while ( v29 )
        {
          sub_CB0380(*(_QWORD *)(v29 + 24));
          v30 = v29;
          v29 = *(_QWORD *)(v29 + 16);
          a2 = 64;
          j_j___libc_free_0(v30, 64);
        }
        v31 = (__int64 *)v11[3];
        v32 = &v31[*((unsigned int *)v11 + 8)];
        if ( v31 != v32 )
        {
          for ( i = v11[3]; ; i = v11[3] )
          {
            v34 = *v31;
            v35 = (unsigned int)(((__int64)v31 - i) >> 3) >> 7;
            a2 = 4096LL << v35;
            if ( v35 >= 0x1E )
              a2 = 0x40000000000LL;
            ++v31;
            sub_C7D6A0(v34, a2, 16);
            if ( v32 == v31 )
              break;
          }
        }
        v36 = (__int64 *)v11[9];
        v19 = &v36[2 * *((unsigned int *)v11 + 20)];
        if ( v36 != v19 )
        {
          do
          {
            a2 = v36[1];
            v37 = *v36;
            v36 += 2;
            sub_C7D6A0(v37, a2, 16);
          }
          while ( v19 != v36 );
          goto LABEL_18;
        }
LABEL_19:
        if ( v19 != v11 + 11 )
          _libc_free(v19, a2);
        v21 = (__int64 *)v11[3];
        if ( v21 != v11 + 5 )
          _libc_free(v21, a2);
        a2 = 160;
        j_j___libc_free_0(v11, 160);
      }
    }
    else
    {
      v10 = *(__int64 ***)(a1 + 592);
      v11 = *v10;
      *v10 = 0;
      if ( v11 )
      {
        v12 = v11[16];
        while ( v12 )
        {
          sub_CB0380(*(_QWORD *)(v12 + 24));
          v13 = v12;
          v12 = *(_QWORD *)(v12 + 16);
          a2 = 64;
          j_j___libc_free_0(v13, 64);
        }
        v14 = (__int64 *)v11[3];
        v15 = &v14[*((unsigned int *)v11 + 8)];
        while ( v15 != v14 )
        {
          v16 = *v14;
          v17 = (unsigned int)(((__int64)v14 - v11[3]) >> 3) >> 7;
          a2 = 4096LL << v17;
          if ( v17 >= 0x1E )
            a2 = 0x40000000000LL;
          ++v14;
          sub_C7D6A0(v16, a2, 16);
        }
        v18 = (__int64 *)v11[9];
        v19 = &v18[2 * *((unsigned int *)v11 + 20)];
        if ( v18 != v19 )
        {
          do
          {
            a2 = v18[1];
            v20 = *v18;
            v18 += 2;
            sub_C7D6A0(v20, a2, 16);
          }
          while ( v19 != v18 );
LABEL_18:
          v19 = (__int64 *)v11[9];
          goto LABEL_19;
        }
        goto LABEL_19;
      }
    }
  }
  v39 = v8;
  sub_CB3D40(a1);
  v38 = sub_CB4460(a1, v39);
  *(_QWORD *)(a1 + 88) = v38;
  *(_QWORD *)(a1 + 672) = v38;
  return 1;
}
