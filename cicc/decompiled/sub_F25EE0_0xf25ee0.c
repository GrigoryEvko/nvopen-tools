// Function: sub_F25EE0
// Address: 0xf25ee0
//
__int64 __fastcall sub_F25EE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 v6; // rdi
  unsigned __int64 v7; // rax
  int v8; // edx
  unsigned __int64 v9; // rax
  _QWORD *v10; // r15
  _QWORD *v11; // rcx
  __int64 *v12; // r13
  unsigned __int64 v13; // r8
  bool v14; // zf
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // rdi
  int v20; // eax
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rsi
  _QWORD *v23; // r15
  __int64 v24; // rax
  _QWORD *v25; // r13
  __int64 result; // rax
  __int64 v27; // r15
  unsigned int v28; // r13d
  unsigned int v29; // esi
  __int64 v30; // rax
  __int64 v31; // [rsp+0h] [rbp-90h]
  unsigned __int64 v32; // [rsp+8h] [rbp-88h]
  _QWORD *v33; // [rsp+10h] [rbp-80h]
  unsigned __int64 v34; // [rsp+10h] [rbp-80h]
  unsigned __int64 v35; // [rsp+18h] [rbp-78h]
  int v36; // [rsp+18h] [rbp-78h]
  _BYTE *v37; // [rsp+20h] [rbp-70h] BYREF
  __int64 v38; // [rsp+28h] [rbp-68h]
  _BYTE v39[96]; // [rsp+30h] [rbp-60h] BYREF

  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(v5 + 48);
  v35 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  v31 = v5 + 48;
  v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 + 48 == (v6 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v9 = 0;
  }
  else
  {
    if ( !v7 )
      goto LABEL_34;
    v8 = *(unsigned __int8 *)(v7 - 24);
    v9 = v7 - 24;
    if ( (unsigned int)(v8 - 30) >= 0xB )
      v9 = 0;
  }
  v10 = (_QWORD *)(*(_QWORD *)(v9 + 24) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v10 == (_QWORD *)v35 )
    goto LABEL_15;
  do
  {
    v11 = v10;
    v12 = v10 - 3;
    v13 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
    v14 = *(v10 - 1) == 0;
    v10 = (_QWORD *)v13;
    if ( !v14 )
    {
      v15 = *(v11 - 2);
      v33 = v11;
      if ( *(_BYTE *)(v15 + 8) == 11 )
        continue;
      v32 = v13;
      v16 = sub_ACADE0((__int64 **)v15);
      sub_F162A0(a1, (__int64)v12, v16);
      *(_BYTE *)(a1 + 240) = 1;
      v11 = v33;
      v13 = v32;
    }
    v17 = (unsigned int)*((unsigned __int8 *)v11 - 24) - 39;
    if ( (unsigned int)v17 <= 0x38 )
    {
      v18 = 0x100060000000001LL;
      if ( _bittest64(&v18, v17) )
        continue;
    }
    if ( *(_BYTE *)(*(v11 - 2) + 8LL) != 11 )
    {
      v34 = v13;
      sub_B44570((__int64)v12);
      sub_F207A0(a1, v12);
      *(_BYTE *)(a1 + 240) = 1;
      v13 = v34;
    }
  }
  while ( v13 != v35 );
  v6 = *(_QWORD *)(v5 + 48);
LABEL_15:
  v19 = v6 & 0xFFFFFFFFFFFFFFF8LL;
  v37 = v39;
  v38 = 0x600000000LL;
  if ( v31 == v19 )
  {
    v21 = 0;
    goto LABEL_19;
  }
  if ( !v19 )
LABEL_34:
    BUG();
  v20 = *(unsigned __int8 *)(v19 - 24);
  v21 = v19 - 24;
  if ( (unsigned int)(v20 - 30) >= 0xB )
    v21 = 0;
LABEL_19:
  v22 = &v37;
  if ( (unsigned __int8)sub_F55920(v21, &v37) )
  {
    v23 = v37;
    v24 = (unsigned int)v38;
    *(_BYTE *)(a1 + 240) = 1;
    v25 = &v23[v24];
    while ( v25 != v23 )
    {
      v22 = (_QWORD *)*v23++;
      sub_F15FC0(*(_QWORD *)(a1 + 40), (__int64)v22);
    }
  }
  result = *(_QWORD *)(v5 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v31 != result )
  {
    if ( !result )
      BUG();
    v27 = result - 24;
    result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
    if ( (unsigned int)result <= 0xA )
    {
      result = sub_B46E30(v27);
      v36 = result;
      if ( (_DWORD)result )
      {
        v28 = 0;
        do
        {
          v29 = v28++;
          v30 = sub_B46EC0(v27, v29);
          v22 = (_QWORD *)v5;
          result = sub_F25920(a1, v5, v30, a3);
        }
        while ( v36 != v28 );
      }
    }
  }
  if ( v37 != v39 )
    return _libc_free(v37, v22);
  return result;
}
