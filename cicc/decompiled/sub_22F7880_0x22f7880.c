// Function: sub_22F7880
// Address: 0x22f7880
//
__int64 __fastcall sub_22F7880(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, unsigned int *a5, _DWORD *a6, __int64 a7)
{
  unsigned int v7; // r15d
  unsigned int v11; // r13d
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v16; // rsi
  unsigned __int64 v17; // r13
  __int64 v18; // rdx
  _WORD *v19; // r14
  size_t v20; // rax
  void (__fastcall *v22)(_BYTE *, __int64, __int64); // rax
  int v23; // r9d
  int v24; // eax
  int v25; // edx
  int v26; // r13d
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 *v32; // r15
  __int64 v33; // [rsp-8h] [rbp-88h]
  unsigned int v34; // [rsp+0h] [rbp-80h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  int v38; // [rsp+10h] [rbp-70h]
  unsigned int v39; // [rsp+1Ch] [rbp-64h]
  unsigned int v40; // [rsp+24h] [rbp-5Ch] BYREF
  __int64 *v41; // [rsp+28h] [rbp-58h] BYREF
  _BYTE v42[16]; // [rsp+30h] [rbp-50h] BYREF
  void (__fastcall *v43)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-40h]
  __int64 v44; // [rsp+48h] [rbp-38h]

  v7 = a4;
  v34 = a4;
  sub_22F3610(a1, a3, &a3[8 * a4], a4, (__int64)a5, (__int64)a6);
  v39 = v7;
  *a6 = 0;
  v40 = 0;
  *a5 = 0;
  v11 = 0;
  if ( !v7 )
    return a1;
  while ( 1 )
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(a1 + 184);
      v19 = *(_WORD **)(v18 + 8LL * v11);
      if ( v19 )
      {
        v20 = strlen(*(const char **)(v18 + 8LL * v11));
        if ( v20 )
          break;
      }
      v40 = ++v11;
      if ( v7 <= v11 )
        return a1;
    }
    if ( *(_BYTE *)(a2 + 50) && v20 == 2 && *v19 == 11565 )
      break;
    if ( *(_BYTE *)(a2 + 49) )
    {
      sub_22F7250((__int64 *)&v41, a2, a1, &v40);
LABEL_8:
      v16 = v41;
      if ( !v41 )
        goto LABEL_20;
      goto LABEL_9;
    }
    v43 = 0;
    v22 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a7 + 16);
    if ( v22 )
    {
      v22(v42, a7, 2);
      v44 = *(_QWORD *)(a7 + 24);
      v43 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a7 + 16);
    }
    sub_22F5A00((__int64 *)&v41, a2, (__int64 (__fastcall ***)(_QWORD))a1, &v40, (__int64)v42);
    if ( !v43 )
      goto LABEL_8;
    v43(v42, v42, 3);
    v16 = v41;
    if ( !v41 )
    {
LABEL_20:
      v23 = v40 + ~v11;
      *a5 = v11;
      *a6 = v23;
      return a1;
    }
LABEL_9:
    v41 = 0;
    sub_22F3A60(a1, v16, v12, v13, v14, v15);
    v17 = (unsigned __int64)v41;
    if ( v41 )
    {
      sub_314D410(v41);
      j_j___libc_free_0(v17);
    }
    v11 = v40;
    if ( v7 <= v40 )
      return a1;
  }
  if ( v34 > ++v40 )
  {
    do
    {
      v24 = sub_22F59B0(a2, *(_DWORD *)(a2 + 64));
      v38 = v25;
      v26 = v24;
      v36 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL * v40);
      v27 = sub_22077B0(0x58u);
      v32 = (__int64 *)v27;
      if ( v27 )
      {
        sub_314D360(v27, v26, v38, (_DWORD)v19, 2, v40, v36, 0);
        v28 = v33;
      }
      sub_22F3A60(a1, v32, v28, v29, v30, v31);
      ++v40;
    }
    while ( v40 < v39 );
  }
  return a1;
}
