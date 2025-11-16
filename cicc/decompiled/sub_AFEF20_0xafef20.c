// Function: sub_AFEF20
// Address: 0xafef20
//
__int64 __fastcall sub_AFEF20(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r14d
  __int64 v6; // rbx
  __int64 v7; // r15
  unsigned __int8 v9; // al
  __int64 *v10; // rcx
  unsigned __int8 v11; // al
  __int64 v12; // rcx
  unsigned __int8 v13; // al
  __int64 v14; // rcx
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  int v17; // r14d
  int v18; // eax
  __int64 v19; // rsi
  _QWORD *v20; // rdi
  unsigned int v21; // eax
  int v22; // r8d
  _QWORD *v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h] BYREF
  __int64 v27; // [rsp+18h] [rbp-68h] BYREF
  __int64 v28; // [rsp+20h] [rbp-60h] BYREF
  int v29; // [rsp+28h] [rbp-58h] BYREF
  __int64 v30; // [rsp+30h] [rbp-50h] BYREF
  int v31; // [rsp+38h] [rbp-48h] BYREF
  int v32[3]; // [rsp+3Ch] [rbp-44h] BYREF
  __int64 v33[7]; // [rsp+48h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = *(_BYTE *)(*a2 - 16);
    if ( (v9 & 2) != 0 )
      v10 = *(__int64 **)(v6 - 32);
    else
      v10 = (__int64 *)(*a2 - 16 - 8LL * ((v9 >> 2) & 0xF));
    v25 = *a2 - 16;
    v26 = *v10;
    v27 = sub_AF5140(v6, 1u);
    v11 = *(_BYTE *)(v6 - 16);
    if ( (v11 & 2) != 0 )
      v12 = *(_QWORD *)(v6 - 32);
    else
      v12 = v25 - 8LL * ((v11 >> 2) & 0xF);
    v28 = *(_QWORD *)(v12 + 16);
    v29 = *(_DWORD *)(v6 + 16);
    v13 = *(_BYTE *)(v6 - 16);
    if ( (v13 & 2) != 0 )
      v14 = *(_QWORD *)(v6 - 32);
    else
      v14 = v25 - 8LL * ((v13 >> 2) & 0xF);
    v30 = *(_QWORD *)(v14 + 24);
    v31 = *(unsigned __int16 *)(v6 + 20);
    v32[0] = *(_DWORD *)(v6 + 24);
    v32[1] = *(_DWORD *)(v6 + 4);
    v15 = *(_BYTE *)(v6 - 16);
    if ( (v15 & 2) != 0 )
      v16 = *(_QWORD *)(v6 - 32);
    else
      v16 = v25 - 8LL * ((v15 >> 2) & 0xF);
    v17 = v4 - 1;
    v33[0] = *(_QWORD *)(v16 + 32);
    v18 = sub_AF8A50(&v26, &v27, &v28, &v29, &v30, &v31, v32, v33);
    v19 = *a2;
    v20 = 0;
    v21 = v17 & v18;
    v22 = 1;
    v23 = (_QWORD *)(v7 + 8LL * v21);
    v24 = *v23;
    if ( *a2 == *v23 )
    {
LABEL_22:
      *a3 = v23;
      return 1;
    }
    else
    {
      while ( v24 != -4096 )
      {
        if ( v24 != -8192 || v20 )
          v23 = v20;
        v21 = v17 & (v22 + v21);
        v24 = *(_QWORD *)(v7 + 8LL * v21);
        if ( v24 == v19 )
        {
          v23 = (_QWORD *)(v7 + 8LL * v21);
          goto LABEL_22;
        }
        ++v22;
        v20 = v23;
        v23 = (_QWORD *)(v7 + 8LL * v21);
      }
      if ( !v20 )
        v20 = v23;
      *a3 = v20;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
