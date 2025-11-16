// Function: sub_AFEC80
// Address: 0xafec80
//
__int64 __fastcall sub_AFEC80(__int64 a1, __int64 *a2, _QWORD *a3)
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
  __int64 v16; // rcx
  unsigned __int8 v17; // al
  __int64 v18; // rcx
  unsigned __int8 v19; // al
  __int64 v20; // rdx
  int v21; // r14d
  int v22; // eax
  __int64 v23; // rsi
  _QWORD *v24; // rdi
  unsigned int v25; // eax
  int v26; // r8d
  _QWORD *v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // [rsp+8h] [rbp-98h]
  __int64 v30; // [rsp+10h] [rbp-90h] BYREF
  __int64 v31; // [rsp+18h] [rbp-88h] BYREF
  __int64 v32; // [rsp+20h] [rbp-80h] BYREF
  __int64 v33; // [rsp+28h] [rbp-78h] BYREF
  int v34; // [rsp+30h] [rbp-70h] BYREF
  __int64 v35; // [rsp+38h] [rbp-68h] BYREF
  __int8 v36; // [rsp+40h] [rbp-60h] BYREF
  __int8 v37[7]; // [rsp+41h] [rbp-5Fh] BYREF
  __int64 v38[2]; // [rsp+48h] [rbp-58h] BYREF
  int v39; // [rsp+58h] [rbp-48h]
  __int64 v40[8]; // [rsp+60h] [rbp-40h] BYREF

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
    v29 = *a2 - 16;
    v30 = *v10;
    v31 = sub_AF5140(v6, 1u);
    v32 = sub_AF5140(v6, 5u);
    v11 = *(_BYTE *)(v6 - 16);
    if ( (v11 & 2) != 0 )
      v12 = *(_QWORD *)(v6 - 32);
    else
      v12 = v29 - 8LL * ((v11 >> 2) & 0xF);
    v33 = *(_QWORD *)(v12 + 16);
    v34 = *(_DWORD *)(v6 + 16);
    v13 = *(_BYTE *)(v6 - 16);
    if ( (v13 & 2) != 0 )
      v14 = *(_QWORD *)(v6 - 32);
    else
      v14 = v29 - 8LL * ((v13 >> 2) & 0xF);
    v35 = *(_QWORD *)(v14 + 24);
    v36 = *(_BYTE *)(v6 + 20);
    v37[0] = *(_BYTE *)(v6 + 21);
    v15 = *(_BYTE *)(v6 - 16);
    if ( (v15 & 2) != 0 )
      v16 = *(_QWORD *)(v6 - 32);
    else
      v16 = v29 - 8LL * ((v15 >> 2) & 0xF);
    v38[0] = *(_QWORD *)(v16 + 48);
    v17 = *(_BYTE *)(v6 - 16);
    if ( (v17 & 2) != 0 )
      v18 = *(_QWORD *)(v6 - 32);
    else
      v18 = v29 - 8LL * ((v17 >> 2) & 0xF);
    v38[1] = *(_QWORD *)(v18 + 56);
    v39 = *(_DWORD *)(v6 + 4);
    v19 = *(_BYTE *)(v6 - 16);
    if ( (v19 & 2) != 0 )
      v20 = *(_QWORD *)(v6 - 32);
    else
      v20 = v29 - 8LL * ((v19 >> 2) & 0xF);
    v21 = v4 - 1;
    v40[0] = *(_QWORD *)(v20 + 64);
    v22 = sub_AF8D50(&v30, &v31, &v32, &v33, &v34, &v35, &v36, v37, v38, v40);
    v23 = *a2;
    v24 = 0;
    v25 = v21 & v22;
    v26 = 1;
    v27 = (_QWORD *)(v7 + 8LL * v25);
    v28 = *v27;
    if ( *a2 == *v27 )
    {
LABEL_28:
      *a3 = v27;
      return 1;
    }
    else
    {
      while ( v28 != -4096 )
      {
        if ( v28 != -8192 || v24 )
          v27 = v24;
        v25 = v21 & (v26 + v25);
        v28 = *(_QWORD *)(v7 + 8LL * v25);
        if ( v28 == v23 )
        {
          v27 = (_QWORD *)(v7 + 8LL * v25);
          goto LABEL_28;
        }
        ++v26;
        v24 = v27;
        v27 = (_QWORD *)(v7 + 8LL * v25);
      }
      if ( !v24 )
        v24 = v27;
      *a3 = v24;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
