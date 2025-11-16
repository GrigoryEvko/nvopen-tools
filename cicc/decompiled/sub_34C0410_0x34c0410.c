// Function: sub_34C0410
// Address: 0x34c0410
//
void __fastcall sub_34C0410(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  _QWORD *v7; // rax
  unsigned __int64 v8; // rbx
  _QWORD *i; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // r14
  __int64 v14; // r12
  __int64 *v15; // r15
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  bool v18; // cf
  unsigned __int64 v19; // rax
  unsigned int v20; // esi
  unsigned __int64 *v21; // r14
  _QWORD *v22; // rdx
  _QWORD *v23; // rax
  unsigned __int64 v24; // rbx
  __int64 v25; // r12
  __int64 v26; // r13
  unsigned __int64 v27; // rdi
  int v28; // eax
  __int64 v29; // rsi
  unsigned __int64 v30; // rdi
  __int64 v31; // [rsp+8h] [rbp-88h]
  __int64 v33; // [rsp+18h] [rbp-78h]
  __int64 v34; // [rsp+20h] [rbp-70h]
  __int64 v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v37; // [rsp+40h] [rbp-50h] BYREF
  __int64 v38; // [rsp+48h] [rbp-48h]
  _QWORD v39[8]; // [rsp+50h] [rbp-40h] BYREF

  v6 = a1;
  v7 = v39;
  v8 = *(unsigned int *)(a2 + 120);
  v37 = v39;
  v38 = 0x200000000LL;
  if ( v8 )
  {
    if ( v8 > 2 )
    {
      sub_C8D5F0((__int64)&v37, v39, v8, 8u, a5, a6);
      v7 = &v37[(unsigned int)v38];
      for ( i = &v37[v8]; i != v7; ++v7 )
      {
LABEL_4:
        if ( v7 )
          *v7 = 0;
      }
    }
    else
    {
      i = &v39[v8];
      if ( i != v39 )
        goto LABEL_4;
    }
    LODWORD(v38) = v8;
  }
  v31 = *(_QWORD *)(a1 + 112);
  if ( *(_QWORD *)(a1 + 104) == v31 )
  {
    v33 = 0;
  }
  else
  {
    v34 = *(_QWORD *)(a1 + 104);
    v33 = 0;
    do
    {
      v10 = *(_QWORD *)(*(_QWORD *)v34 + 8LL);
      v36 = sub_2F06CB0(*(_QWORD *)(v6 + 232), v10);
      v11 = v33 + v36;
      if ( __CFADD__(v33, v36) )
        v11 = -1;
      v33 = v11;
      v12 = *(unsigned int *)(a2 + 120);
      if ( (unsigned int)v12 > 1 )
      {
        v13 = v37;
        v35 = *(_QWORD *)(a2 + 112) + 8 * v12;
        v14 = v6;
        v15 = *(__int64 **)(a2 + 112);
        do
        {
          v16 = sub_2E441D0(*(_QWORD *)(v14 + 240), v10, *v15);
          v17 = sub_1098D20((unsigned __int64 *)&v36, v16);
          v18 = __CFADD__(*v13, v17);
          v19 = *v13 + v17;
          if ( v18 )
            *v13 = -1;
          else
            *v13 = v19;
          ++v15;
          ++v13;
        }
        while ( v15 != (__int64 *)v35 );
        v6 = v14;
      }
      v34 += 16;
    }
    while ( v31 != v34 );
  }
  sub_2F06FC0(*(_QWORD *)(v6 + 232), a2, v33);
  v20 = *(_DWORD *)(a2 + 120);
  if ( v20 <= 1 )
  {
    v30 = (unsigned __int64)v37;
    if ( v37 != v39 )
      goto LABEL_31;
  }
  else
  {
    v21 = v37;
    v22 = &v37[(unsigned int)v38];
    if ( v37 != v22 )
    {
      v23 = v37;
      v24 = 0;
      do
      {
        v18 = __CFADD__(*v23, v24);
        v24 += *v23;
        if ( v18 )
          v24 = -1;
        ++v23;
      }
      while ( v23 != v22 );
      if ( v24 )
      {
        v25 = *(_QWORD *)(a2 + 112);
        v26 = v25 + 8LL * v20;
        do
        {
          v27 = *v21++;
          v28 = sub_F02DD0(v27, v24);
          v29 = v25;
          v25 += 8;
          sub_2E32F90(a2, v29, v28);
        }
        while ( v25 != v26 );
        v21 = v37;
      }
    }
    if ( v21 != v39 )
    {
      v30 = (unsigned __int64)v21;
LABEL_31:
      _libc_free(v30);
    }
  }
}
