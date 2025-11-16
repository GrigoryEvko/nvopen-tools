// Function: sub_2B1D170
// Address: 0x2b1d170
//
__int64 __fastcall sub_2B1D170(_QWORD *a1, _DWORD *a2)
{
  __int64 *v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rax
  unsigned int v7; // edx
  _QWORD *v8; // rcx
  _DWORD *v9; // r8
  int v10; // esi
  int v11; // eax
  __int64 v12; // rax
  int v13; // edx
  int v14; // r8d
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 result; // rax
  signed __int64 v18; // rax
  int v19; // edx
  bool v20; // zf
  __int64 v21; // rdx
  __int64 v22; // rdx
  unsigned __int64 *v23; // rsi
  bool v24; // of
  __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  int v29; // ecx
  int v30; // r10d
  bool v31; // cc
  unsigned __int64 v32; // rcx
  unsigned __int64 v33; // rcx
  __int64 v34; // [rsp+0h] [rbp-20h]

  v4 = *(__int64 **)(**(_QWORD **)a2 + 8LL);
  v5 = *(_QWORD *)(*a1 + 3528LL);
  v6 = *(unsigned int *)(*a1 + 3544LL);
  v34 = (__int64)v4;
  if ( (_DWORD)v6 )
  {
    v7 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (_QWORD *)(v5 + 24LL * v7);
    v9 = (_DWORD *)*v8;
    if ( a2 == (_DWORD *)*v8 )
    {
LABEL_3:
      if ( v8 != (_QWORD *)(v5 + 24 * v6) )
      {
        v34 = sub_BCCE00((_QWORD *)*v4, v8[1]);
        v4 = (__int64 *)v34;
      }
    }
    else
    {
      v29 = 1;
      while ( v9 != (_DWORD *)-4096LL )
      {
        v30 = v29 + 1;
        v7 = (v6 - 1) & (v29 + v7);
        v8 = (_QWORD *)(v5 + 24LL * v7);
        v9 = (_DWORD *)*v8;
        if ( a2 == (_DWORD *)*v8 )
          goto LABEL_3;
        v29 = v30;
      }
    }
  }
  v10 = a2[30];
  if ( !v10 )
    v10 = a2[2];
  v11 = *((unsigned __int8 *)v4 + 8);
  if ( (_BYTE)v11 == 17 )
  {
    v10 *= *((_DWORD *)v4 + 8);
LABEL_9:
    v4 = *(__int64 **)v4[2];
    goto LABEL_10;
  }
  if ( (unsigned int)(v11 - 17) <= 1 )
    goto LABEL_9;
LABEL_10:
  sub_BCDA70(v4, v10);
  v12 = sub_DFDD20(*(_QWORD *)(*a1 + 3296LL));
  v14 = v13;
  v15 = a1[1];
  if ( v14 == 1 )
    *(_DWORD *)(v15 + 8) = 1;
  if ( __OFADD__(*(_QWORD *)v15, v12) )
  {
    v31 = v12 <= 0;
    v16 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v31 )
      v16 = 0x8000000000000000LL;
  }
  else
  {
    v16 = *(_QWORD *)v15 + v12;
  }
  *(_QWORD *)v15 = v16;
  result = (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17;
  if ( (unsigned int)result <= 1 )
  {
    v18 = sub_DFDD20(*(_QWORD *)(*a1 + 3296LL));
    v20 = v19 == 1;
    v21 = (unsigned int)a2[2];
    if ( v20 )
    {
      v27 = v18 * v21;
      if ( is_mul_ok(v18, (unsigned int)a2[2]) )
      {
        v23 = (unsigned __int64 *)a1[1];
        result = v27;
        v28 = *v23;
        *((_DWORD *)v23 + 2) = 1;
        v24 = __OFSUB__(v28, result);
        v25 = v28 - result;
        if ( !v24 )
          goto LABEL_18;
LABEL_25:
        v26 = 0x8000000000000000LL;
        if ( result > 0 )
          goto LABEL_19;
        goto LABEL_26;
      }
      v23 = (unsigned __int64 *)a1[1];
      if ( a2[2] && v18 > 0 )
      {
        v32 = *v23;
        *((_DWORD *)v23 + 2) = 1;
        result = 0x7FFFFFFFFFFFFFFFLL;
        v24 = __OFSUB__(v32, 0x7FFFFFFFFFFFFFFFLL);
        v26 = v32 - 0x7FFFFFFFFFFFFFFFLL;
        if ( v24 )
          v26 = 0x8000000000000000LL;
        goto LABEL_19;
      }
      v33 = *v23;
      *((_DWORD *)v23 + 2) = 1;
      result = 0x8000000000000000LL;
      v24 = __OFSUB__(v33, 0x8000000000000000LL);
      v26 = v33 + 0x8000000000000000LL;
      if ( !v24 )
      {
LABEL_19:
        *v23 = v26;
        return result;
      }
    }
    else
    {
      v22 = v18 * v21;
      if ( is_mul_ok(v18, (unsigned int)a2[2]) )
      {
        v23 = (unsigned __int64 *)a1[1];
        result = v22;
        v24 = __OFSUB__(*v23, v22);
        v25 = *v23 - v22;
        if ( !v24 )
        {
LABEL_18:
          v26 = v25;
          goto LABEL_19;
        }
        goto LABEL_25;
      }
      v23 = (unsigned __int64 *)a1[1];
      if ( a2[2] && v18 > 0 )
      {
        result = 0x7FFFFFFFFFFFFFFFLL;
        v26 = 0x8000000000000000LL;
        v25 = *v23 - 0x7FFFFFFFFFFFFFFFLL;
        if ( __OFSUB__(*v23, 0x7FFFFFFFFFFFFFFFLL) )
          goto LABEL_19;
        goto LABEL_18;
      }
      result = 0x8000000000000000LL;
      v26 = *v23 + 0x8000000000000000LL;
      if ( !__OFSUB__(*v23, 0x8000000000000000LL) )
        goto LABEL_19;
    }
LABEL_26:
    v26 = 0x7FFFFFFFFFFFFFFFLL;
    goto LABEL_19;
  }
  return result;
}
