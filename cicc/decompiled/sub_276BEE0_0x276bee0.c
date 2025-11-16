// Function: sub_276BEE0
// Address: 0x276bee0
//
unsigned __int64 *__fastcall sub_276BEE0(unsigned __int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // r14
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rsi
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r15
  bool v12; // zf
  __int64 *v13; // r8
  __int64 *v14; // r13
  _QWORD *v15; // rdx
  unsigned int v16; // esi
  __int64 v17; // rsi
  char v18; // dl
  unsigned __int64 v19; // r13
  unsigned __int64 i; // r15
  unsigned int v21; // esi
  __int64 v22; // rdx
  unsigned int v23; // edx
  unsigned __int64 j; // r12
  unsigned __int64 v25; // rdi
  unsigned __int64 *v26; // rdi
  unsigned __int64 v28; // r15
  __int64 v29; // rax
  _QWORD *v30; // [rsp+0h] [rbp-60h]
  _QWORD *v31; // [rsp+8h] [rbp-58h]
  unsigned __int64 v32; // [rsp+10h] [rbp-50h]
  unsigned __int64 v34; // [rsp+20h] [rbp-40h]
  unsigned __int64 v35; // [rsp+28h] [rbp-38h]

  v3 = a2;
  v5 = a1[1];
  v35 = *a1;
  v6 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(v5 - *a1) >> 4);
  if ( v6 == 0x124924924924924LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(v5 - *a1) >> 4);
  v8 = __CFADD__(v7, v6);
  v9 = v7 + v6;
  v10 = a2 - v35;
  if ( v8 )
  {
    v28 = 0x7FFFFFFFFFFFFFC0LL;
  }
  else
  {
    if ( !v9 )
    {
      v32 = 0;
      v11 = 112;
      v34 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x124924924924924LL )
      v9 = 0x124924924924924LL;
    v28 = 112 * v9;
  }
  v30 = a3;
  v29 = sub_22077B0(v28);
  v10 = a2 - v35;
  a3 = v30;
  v34 = v29;
  v32 = v29 + v28;
  v11 = v29 + 112;
LABEL_7:
  v12 = v34 + v10 == 0;
  v13 = (__int64 *)(v34 + v10);
  v14 = v13;
  if ( !v12 )
  {
    v31 = a3;
    sub_276A0C0(v13, a3);
    v15 = v31;
    v16 = *((_DWORD *)v31 + 22);
    *((_DWORD *)v14 + 22) = v16;
    if ( v16 > 0x40 )
    {
      sub_C43780((__int64)(v14 + 10), (const void **)v31 + 10);
      v15 = v31;
    }
    else
    {
      v14[10] = v31[10];
    }
    v17 = v15[12];
    v18 = *((_BYTE *)v15 + 104);
    v14[12] = v17;
    *((_BYTE *)v14 + 104) = v18;
  }
  v19 = v35;
  if ( a2 != v35 )
  {
    for ( i = v34; ; i += 112LL )
    {
      if ( i )
      {
        sub_276A0C0((__int64 *)i, (_QWORD *)v19);
        v21 = *(_DWORD *)(v19 + 88);
        *(_DWORD *)(i + 88) = v21;
        if ( v21 <= 0x40 )
          *(_QWORD *)(i + 80) = *(_QWORD *)(v19 + 80);
        else
          sub_C43780(i + 80, (const void **)(v19 + 80));
        *(_QWORD *)(i + 96) = *(_QWORD *)(v19 + 96);
        *(_BYTE *)(i + 104) = *(_BYTE *)(v19 + 104);
      }
      v19 += 112LL;
      if ( a2 == v19 )
        break;
    }
    v11 = i + 224;
  }
  if ( a2 != v5 )
  {
    do
    {
      sub_276A0C0((__int64 *)v11, (_QWORD *)v3);
      v23 = *(_DWORD *)(v3 + 88);
      *(_DWORD *)(v11 + 88) = v23;
      if ( v23 <= 0x40 )
        *(_QWORD *)(v11 + 80) = *(_QWORD *)(v3 + 80);
      else
        sub_C43780(v11 + 80, (const void **)(v3 + 80));
      v22 = *(_QWORD *)(v3 + 96);
      v3 += 112;
      v11 += 112;
      *(_QWORD *)(v11 - 16) = v22;
      *(_BYTE *)(v11 - 8) = *(_BYTE *)(v3 - 8);
    }
    while ( v5 != v3 );
  }
  for ( j = v35; v5 != j; j += 112LL )
  {
    if ( *(_DWORD *)(j + 88) > 0x40u )
    {
      v25 = *(_QWORD *)(j + 80);
      if ( v25 )
        j_j___libc_free_0_0(v25);
    }
    v26 = (unsigned __int64 *)j;
    sub_2767770(v26);
  }
  if ( v35 )
    j_j___libc_free_0(v35);
  *a1 = v34;
  a1[1] = v11;
  a1[2] = v32;
  return a1;
}
