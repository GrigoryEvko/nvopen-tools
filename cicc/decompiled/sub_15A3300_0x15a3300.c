// Function: sub_15A3300
// Address: 0x15a3300
//
__int64 __fastcall sub_15A3300(unsigned __int64 a1, __int64 a2, char a3)
{
  unsigned __int64 v3; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // r8
  unsigned __int64 v9; // rax
  int v11; // eax
  __int64 v12; // rax
  __int64 *v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r15
  __int64 **v18; // r12
  __int64 v19; // rax
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  __int64 **v22; // rbx
  unsigned __int64 v23; // r8
  __int64 **v24; // rax
  int v25; // ecx
  __int64 **v26; // rdx
  unsigned int v27; // ecx
  unsigned __int8 v28; // r8
  unsigned __int8 v29; // r8
  __int64 v30; // r12
  unsigned __int64 v31; // [rsp+8h] [rbp-98h]
  int v32; // [rsp+18h] [rbp-88h] BYREF
  char v33; // [rsp+1Ch] [rbp-84h]
  __int64 **v34; // [rsp+20h] [rbp-80h] BYREF
  __int64 v35; // [rsp+28h] [rbp-78h]
  _BYTE v36[112]; // [rsp+30h] [rbp-70h] BYREF

  v3 = a1;
  v5 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
    v5 = **(_QWORD **)(v5 + 16);
  v6 = a2;
  if ( *(_BYTE *)(a2 + 8) == 16 )
    v6 = **(_QWORD **)(a2 + 16);
  v7 = *(_QWORD *)(v5 + 24);
  if ( *(_QWORD *)(v6 + 24) != v7 )
  {
    v8 = sub_1646BA0(v7, *(_DWORD *)(v6 + 8) >> 8);
    if ( *(_BYTE *)(a2 + 8) != 16 )
    {
LABEL_8:
      v9 = sub_15A3300(v3, v8, 0);
      return sub_15A2980(0x2Fu, v9, (__int64 **)a2, a3);
    }
LABEL_7:
    v8 = sub_16463B0(v8, *(unsigned int *)(a2 + 32));
    goto LABEL_8;
  }
  if ( *(_BYTE *)(v3 + 16) != 5 )
    return sub_15A2980(0x30u, v3, (__int64 **)a2, a3);
  v11 = *(unsigned __int16 *)(v3 + 18);
  if ( v11 != 32 )
  {
    if ( v11 == 47 )
    {
      v3 = *(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
      v12 = *(_QWORD *)v3;
      if ( *(_BYTE *)(*(_QWORD *)v3 + 8LL) == 16 )
        v12 = **(_QWORD **)(v12 + 16);
      v8 = sub_1646BA0(*(_QWORD *)(v12 + 24), *(_DWORD *)(v6 + 8) >> 8);
      if ( *(_BYTE *)(a2 + 8) != 16 )
        goto LABEL_8;
      goto LABEL_7;
    }
    return sub_15A2980(0x30u, v3, (__int64 **)a2, a3);
  }
  v13 = *(__int64 **)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
  v14 = *v13;
  if ( *(_BYTE *)(*v13 + 8) == 16 )
    v14 = **(_QWORD **)(v14 + 16);
  v15 = sub_1646BA0(*(_QWORD *)(v14 + 24), *(_DWORD *)(v6 + 8) >> 8);
  v16 = v15;
  if ( *(_BYTE *)(a2 + 8) == 16 )
    v16 = sub_16463B0(v15, *(_QWORD *)(a2 + 32));
  v17 = sub_15A3300(v13, v16, 0);
  if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
  {
    v20 = *(_QWORD *)(v3 - 8);
    v19 = 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
    v18 = (__int64 **)(v20 + v19);
  }
  else
  {
    v18 = (__int64 **)v3;
    v19 = 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
    v20 = v3 - v19;
  }
  v21 = v19 - 24;
  v22 = (__int64 **)(v20 + 24);
  v34 = (__int64 **)v36;
  v35 = 0x800000000LL;
  v23 = 0xAAAAAAAAAAAAAAABLL * (v21 >> 3);
  if ( (unsigned __int64)v21 > 0xC0 )
  {
    v31 = 0xAAAAAAAAAAAAAAABLL * (v21 >> 3);
    sub_16CD150(&v34, v36, v31, 8);
    v26 = v34;
    v25 = v35;
    LODWORD(v23) = v31;
    v24 = &v34[(unsigned int)v35];
  }
  else
  {
    v24 = (__int64 **)v36;
    v25 = 0;
    v26 = (__int64 **)v36;
  }
  if ( v22 != v18 )
  {
    do
    {
      if ( v24 )
        *v24 = *v22;
      v22 += 3;
      ++v24;
    }
    while ( v22 != v18 );
    v26 = v34;
    v25 = v35;
  }
  v27 = v23 + v25;
  v28 = *(_BYTE *)(v3 + 17);
  LODWORD(v35) = v27;
  v29 = v28 >> 1;
  if ( (int)v29 >> 1 )
  {
    v33 = 1;
    v32 = ((int)v29 >> 1) - 1;
  }
  else
  {
    v33 = 0;
  }
  v30 = sub_15A2E80(0, v17, v26, v27, v29 & 1, (__int64)&v32, 0);
  if ( v34 != (__int64 **)v36 )
    _libc_free((unsigned __int64)v34);
  return v30;
}
