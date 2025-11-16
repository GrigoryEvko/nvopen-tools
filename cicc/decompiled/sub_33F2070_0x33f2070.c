// Function: sub_33F2070
// Address: 0x33f2070
//
_QWORD *__fastcall sub_33F2070(unsigned int a1, __int64 a2, char *a3, __int64 a4, _QWORD *a5)
{
  char *v9; // r8
  __int64 v10; // rsi
  __int64 v11; // rdx
  char *v12; // rax
  char *v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // r15
  unsigned int v16; // r8d
  __int64 i; // rdi
  __int64 v18; // rax
  __int64 *v19; // rcx
  __int64 v20; // r13
  unsigned int v21; // r14d
  __int64 v22; // rax
  __int64 v24; // rcx
  int v25; // eax
  __int64 v26; // r8
  int v27; // eax
  __int64 v28; // rax
  _QWORD *v29; // rbx
  __int64 v30; // [rsp+10h] [rbp-50h]
  unsigned int v31; // [rsp+18h] [rbp-48h]
  unsigned int v32; // [rsp+1Ch] [rbp-44h]
  __int64 v33; // [rsp+20h] [rbp-40h] BYREF
  int v34; // [rsp+28h] [rbp-38h]

  v9 = &a3[16 * a4];
  v10 = (16 * a4) >> 6;
  v11 = (16 * a4) >> 4;
  if ( v10 > 0 )
  {
    v12 = a3;
    v13 = &a3[64 * v10];
    while ( *(_DWORD *)(*(_QWORD *)v12 + 24LL) == 51 )
    {
      if ( *(_DWORD *)(*((_QWORD *)v12 + 2) + 24LL) != 51 )
      {
        v12 += 16;
        goto LABEL_8;
      }
      if ( *(_DWORD *)(*((_QWORD *)v12 + 4) + 24LL) != 51 )
      {
        v12 += 32;
        goto LABEL_8;
      }
      if ( *(_DWORD *)(*((_QWORD *)v12 + 6) + 24LL) != 51 )
      {
        v12 += 48;
        goto LABEL_8;
      }
      v12 += 64;
      if ( v12 == v13 )
      {
        v11 = (v9 - v12) >> 4;
        goto LABEL_28;
      }
    }
    goto LABEL_8;
  }
  v12 = a3;
LABEL_28:
  if ( v11 == 2 )
    goto LABEL_40;
  if ( v11 == 3 )
  {
    if ( *(_DWORD *)(*(_QWORD *)v12 + 24LL) != 51 )
      goto LABEL_8;
    v12 += 16;
LABEL_40:
    if ( *(_DWORD *)(*(_QWORD *)v12 + 24LL) != 51 )
      goto LABEL_8;
    v12 += 16;
    goto LABEL_31;
  }
  if ( v11 != 1 )
    goto LABEL_32;
LABEL_31:
  if ( *(_DWORD *)(*(_QWORD *)v12 + 24LL) == 51 )
    goto LABEL_32;
LABEL_8:
  if ( v9 == v12 )
  {
LABEL_32:
    v33 = 0;
    v34 = 0;
    v29 = sub_33F17F0(a5, 51, (__int64)&v33, a1, a2);
    if ( v33 )
      sub_B91220((__int64)&v33, v33);
    return v29;
  }
  if ( (_DWORD)a4 )
  {
    v14 = (unsigned int)(a4 - 1);
    v15 = 0;
    v16 = 0;
    for ( i = 0; ; i = v20 )
    {
      v18 = *(_QWORD *)&a3[16 * v15];
      if ( *(_DWORD *)(v18 + 24) != 158 )
        break;
      v19 = *(__int64 **)(v18 + 40);
      v20 = *v19;
      v21 = *((_DWORD *)v19 + 2);
      v22 = *(_QWORD *)(*v19 + 48) + 16LL * v21;
      if ( *(_WORD *)v22 != (_WORD)a1 || *(_QWORD *)(v22 + 8) != a2 && !(_WORD)a1 )
        break;
      if ( i && (v21 != v16 || v20 != i) )
        break;
      v24 = v19[5];
      v25 = *(_DWORD *)(v24 + 24);
      if ( v25 != 11 && v25 != 35 )
        break;
      v26 = *(_QWORD *)(v24 + 96);
      v32 = *(_DWORD *)(v26 + 32);
      if ( v32 <= 0x40 )
      {
        v28 = *(_QWORD *)(v26 + 24);
      }
      else
      {
        v31 = a1;
        v30 = *(_QWORD *)(v24 + 96);
        v27 = sub_C444A0(v26 + 24);
        a1 = v31;
        if ( v32 - v27 > 0x40 )
          return 0;
        v28 = **(_QWORD **)(v30 + 24);
      }
      if ( v15 != v28 )
        break;
      if ( v15 == v14 )
        return (_QWORD *)v20;
      ++v15;
      v16 = v21;
    }
    return 0;
  }
  else
  {
    return 0;
  }
}
