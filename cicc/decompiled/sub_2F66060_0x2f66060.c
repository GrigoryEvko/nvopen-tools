// Function: sub_2F66060
// Address: 0x2f66060
//
__int64 __fastcall sub_2F66060(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v4; // eax
  unsigned int v5; // r12d
  int v6; // eax
  int v7; // edx
  unsigned int v8; // ecx
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // rdx
  __int64 *v12; // r13
  int v13; // edx
  __int64 v14; // rax
  int v15; // eax
  unsigned int v17; // eax
  __int64 v18; // r14
  unsigned int v19; // edx
  unsigned __int64 v20; // r14
  __int64 v21; // rdi
  __int64 v22; // rax
  int v23; // edx
  unsigned int v24; // ecx
  unsigned int v25; // esi
  __int64 v26; // rdi
  unsigned int v27; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v28; // [rsp+4h] [rbp-2Ch] BYREF
  int v29[10]; // [rsp+8h] [rbp-28h] BYREF

  *(_WORD *)((char *)a1 + 25) = 0;
  *((_DWORD *)a1 + 2) = 0;
  *((_DWORD *)a1 + 3) = 0;
  a1[2] = 0;
  a1[4] = 0;
  v3 = (__int64)*a1;
  v27 = 0;
  v28 = 0;
  *(_QWORD *)v29 = 0;
  v4 = sub_2F61710(v3, a2, &v27, &v28, v29, &v29[1]);
  if ( !(_BYTE)v4 )
    return 0;
  v5 = v4;
  v6 = v29[0];
  v7 = v29[1];
  v8 = v27;
  *((_BYTE *)a1 + 24) = *(_QWORD *)v29 != 0;
  if ( v8 - 1 <= 0x3FFFFFFE )
  {
    if ( v28 - 1 <= 0x3FFFFFFE )
      return 0;
    v27 = v28;
    v28 = v8;
    *(_QWORD *)v29 = __PAIR64__(v6, v7);
    *((_BYTE *)a1 + 26) = 1;
  }
  v9 = sub_2E88D60(a2);
  v10 = v28;
  v11 = *(_QWORD *)(*(_QWORD *)(v9 + 32) + 56LL);
  v12 = (__int64 *)(*(_QWORD *)(v11 + 16LL * (v27 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v28 - 1 > 0x3FFFFFFE )
  {
    v18 = *(_QWORD *)(v11 + 16LL * (v28 & 0x7FFFFFFF));
    v19 = v29[0];
    v20 = v18 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v29[0] )
    {
      if ( v29[1] )
      {
        if ( v29[0] != v29[1] && v28 == v27 )
          return 0;
        v22 = sub_2FF69E0((unsigned int)*a1, (_DWORD)v12, v29[0], v20, v29[1], (int)a1 + 20, (__int64)(a1 + 2));
        a1[4] = (_QWORD *)v22;
        if ( !v22 )
          return 0;
        goto LABEL_20;
      }
      v26 = (__int64)*a1;
      *((_DWORD *)a1 + 4) = v29[0];
      v22 = (*(__int64 (__fastcall **)(__int64, __int64 *, unsigned __int64, _QWORD))(*(_QWORD *)v26 + 256LL))(
              v26,
              v12,
              v20,
              v19);
      a1[4] = (_QWORD *)v22;
    }
    else
    {
      v21 = (__int64)*a1;
      if ( v29[1] )
      {
        *((_DWORD *)a1 + 5) = v29[1];
        v22 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64 *))(*(_QWORD *)v21 + 256LL))(v21, v20, v12);
      }
      else
      {
        v22 = sub_2FF6970(v21, v20, v12, (unsigned int)v29[1]);
      }
      a1[4] = (_QWORD *)v22;
    }
    if ( !v22 )
      return 0;
LABEL_20:
    v23 = *((_DWORD *)a1 + 4);
    if ( v23 && !*((_DWORD *)a1 + 5) )
    {
      v24 = v27;
      v25 = v28;
      *((_DWORD *)a1 + 5) = v23;
      *((_BYTE *)a1 + 26) ^= 1u;
      v27 = v25;
      v28 = v24;
      *((_DWORD *)a1 + 4) = 0;
    }
    *((_BYTE *)a1 + 25) = v20 != v22 || v12 != (__int64 *)v22;
    goto LABEL_10;
  }
  if ( v29[1] )
  {
    v17 = sub_E91CF0(*a1, v28, v29[1]);
    v28 = v17;
    v10 = v17;
    if ( !v17 )
      return 0;
    v13 = v29[0];
    v29[1] = 0;
    if ( !v29[0] )
    {
      if ( v17 - 1 > 0x3FFFFFFE )
        return 0;
      goto LABEL_8;
    }
  }
  else
  {
    v13 = v29[0];
    if ( !v29[0] )
    {
LABEL_8:
      v14 = v10 >> 3;
      if ( (unsigned int)v14 < *(unsigned __int16 *)(*v12 + 22) )
      {
        v15 = *(unsigned __int8 *)(*(_QWORD *)(*v12 + 8) + v14);
        if ( _bittest(&v15, v10 & 7) )
          goto LABEL_10;
      }
      return 0;
    }
  }
  v28 = sub_E91D60(*a1, v10, v13, *v12);
  if ( v28 )
  {
LABEL_10:
    *((_DWORD *)a1 + 3) = v27;
    *((_DWORD *)a1 + 2) = v28;
    return v5;
  }
  return 0;
}
