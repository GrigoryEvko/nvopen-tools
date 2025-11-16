// Function: sub_D69810
// Address: 0xd69810
//
_BYTE *__fastcall sub_D69810(
        _BYTE *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 (__fastcall *a5)(__int64, _QWORD),
        __int64 a6)
{
  _BYTE *v6; // r14
  int v10; // esi
  int v11; // esi
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // r9
  _BYTE *result; // rax
  _BYTE **v17; // rax
  __int64 v18; // rsi
  int v19; // ecx
  int v20; // ecx
  unsigned int v21; // edx
  _QWORD *v22; // rax
  _BYTE *v23; // rdi
  int v24; // eax
  int v25; // eax
  int v26; // r8d
  int v27; // r8d
  __int64 v30; // [rsp+18h] [rbp-68h]
  __int64 v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+28h] [rbp-58h] BYREF
  unsigned __int64 v33[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v34; // [rsp+40h] [rbp-40h]

  v6 = a1;
  if ( *a1 == 27 )
  {
    while ( *(_BYTE **)(a4 + 128) != v6 )
    {
      v30 = *((_QWORD *)v6 + 9);
      if ( !a5(a6, *(_QWORD *)(v30 + 40)) )
        break;
      v32 = v30;
      sub_D696B0(v33, a2, &v32);
      if ( v34 )
      {
        v31 = v34;
        sub_D68D70(v33);
        v10 = *(_DWORD *)(a4 + 56);
        if ( v10 )
        {
          v11 = v10 - 1;
          v12 = *(_QWORD *)(a4 + 40);
          v13 = v11 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          v14 = (__int64 *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( v31 == *v14 )
          {
LABEL_7:
            result = (_BYTE *)v14[1];
            if ( result && *result != 26 )
              return result;
          }
          else
          {
            v25 = 1;
            while ( v15 != -4096 )
            {
              v26 = v25 + 1;
              v13 = v11 & (v25 + v13);
              v14 = (__int64 *)(v12 + 16LL * v13);
              v15 = *v14;
              if ( v31 == *v14 )
                goto LABEL_7;
              v25 = v26;
            }
          }
        }
      }
      else
      {
        sub_D68D70(v33);
      }
      v17 = (_BYTE **)(v6 - 64);
      if ( *v6 == 26 )
        v17 = (_BYTE **)(v6 - 32);
      v6 = *v17;
      if ( **v17 != 27 )
        goto LABEL_12;
    }
    return v6;
  }
LABEL_12:
  if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
  {
    v18 = a3 + 16;
    v19 = 3;
  }
  else
  {
    v20 = *(_DWORD *)(a3 + 24);
    v18 = *(_QWORD *)(a3 + 16);
    if ( !v20 )
      return v6;
    v19 = v20 - 1;
  }
  v21 = v19 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v22 = (_QWORD *)(v18 + 16LL * v21);
  v23 = (_BYTE *)*v22;
  if ( v6 != (_BYTE *)*v22 )
  {
    v24 = 1;
    while ( v23 != (_BYTE *)-4096LL )
    {
      v27 = v24 + 1;
      v21 = v19 & (v24 + v21);
      v22 = (_QWORD *)(v18 + 16LL * v21);
      v23 = (_BYTE *)*v22;
      if ( (_BYTE *)*v22 == v6 )
        goto LABEL_17;
      v24 = v27;
    }
    return v6;
  }
LABEL_17:
  result = (_BYTE *)v22[1];
  if ( !result )
    return v6;
  return result;
}
