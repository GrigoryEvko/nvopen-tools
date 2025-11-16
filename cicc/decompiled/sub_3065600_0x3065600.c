// Function: sub_3065600
// Address: 0x3065600
//
unsigned __int64 __fastcall sub_3065600(__int64 a1, _BYTE **a2, int a3, __int64 *a4, __int64 a5, __int64 a6)
{
  _BYTE **v7; // rbx
  __int64 v8; // rcx
  unsigned __int64 v9; // r13
  __int64 v10; // r14
  __int64 *v11; // r15
  unsigned __int8 v12; // al
  __int64 v13; // r12
  _BYTE *v14; // rsi
  int v15; // edi
  __int64 v16; // r8
  __int64 v17; // rdx
  char *v18; // rax
  char v19; // dl
  int v20; // eax
  unsigned int v22; // eax
  unsigned __int64 v23; // rdx
  bool v24; // zf
  unsigned __int64 v25; // rax
  int v26; // edx
  signed __int64 v27; // r12
  int v28; // eax
  bool v29; // of
  int v30; // [rsp+10h] [rbp-A0h]
  int v32; // [rsp+2Ch] [rbp-84h]
  unsigned __int64 v33; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-78h]
  __int64 v35; // [rsp+40h] [rbp-70h] BYREF
  char *v36; // [rsp+48h] [rbp-68h]
  __int64 v37; // [rsp+50h] [rbp-60h]
  int v38; // [rsp+58h] [rbp-58h]
  unsigned __int8 v39; // [rsp+5Ch] [rbp-54h]
  char v40; // [rsp+60h] [rbp-50h] BYREF

  v35 = 0;
  v36 = &v40;
  v37 = 4;
  v38 = 0;
  v39 = 1;
  if ( !a3 )
    return 0;
  v7 = a2;
  v8 = 1;
  v9 = 0;
  v32 = 0;
  v10 = (__int64)&a2[(unsigned int)(a3 - 1) + 1];
  v11 = a4;
  do
  {
    while ( 1 )
    {
      v13 = *v11;
      v14 = *v7;
      v15 = *(unsigned __int8 *)(*v11 + 8);
      v16 = (unsigned int)(v15 - 17);
      v17 = *(unsigned __int8 *)(*v11 + 8);
      if ( (unsigned int)v16 > 1 )
      {
        if ( (_BYTE)v15 == 12 )
          goto LABEL_8;
LABEL_12:
        v12 = *(_BYTE *)(*v11 + 8);
        if ( v15 == 17 )
          v12 = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
        goto LABEL_5;
      }
      v12 = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
      if ( v12 == 12 )
        goto LABEL_8;
      if ( v15 != 18 )
        goto LABEL_12;
LABEL_5:
      if ( v12 > 3u && v12 != 5 && (v12 & 0xFD) != 4 )
        break;
LABEL_8:
      if ( *v14 > 0x15u )
        goto LABEL_18;
LABEL_9:
      ++v7;
      ++v11;
      if ( (_BYTE **)v10 == v7 )
        goto LABEL_27;
    }
    if ( (unsigned int)v16 <= 1 )
      v17 = *(unsigned __int8 *)(**(_QWORD **)(v13 + 16) + 8LL);
    if ( (_BYTE)v17 != 14 || *v14 <= 0x15u )
      goto LABEL_9;
LABEL_18:
    if ( !(_BYTE)v8 )
      goto LABEL_23;
    v18 = v36;
    v17 = (__int64)&v36[8 * HIDWORD(v37)];
    if ( v36 != (char *)v17 )
    {
      while ( v14 != *(_BYTE **)v18 )
      {
        v18 += 8;
        if ( (char *)v17 == v18 )
          goto LABEL_22;
      }
      goto LABEL_9;
    }
LABEL_22:
    if ( HIDWORD(v37) < (unsigned int)v37 )
    {
      ++HIDWORD(v37);
      *(_QWORD *)v17 = v14;
      v20 = *(unsigned __int8 *)(v13 + 8);
      ++v35;
      v8 = v39;
      if ( v20 == 17 )
      {
LABEL_31:
        v22 = *(_DWORD *)(v13 + 32);
        v34 = v22;
        if ( v22 > 0x40 )
        {
          sub_C43690((__int64)&v33, -1, 1);
        }
        else
        {
          v23 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v22;
          v24 = v22 == 0;
          v25 = 0;
          if ( !v24 )
            v25 = v23;
          v33 = v25;
        }
        v27 = sub_3064F80(a1, v13, (__int64 *)&v33, 0, 1);
        if ( v34 > 0x40 && v33 )
        {
          v30 = v26;
          j_j___libc_free_0_0(v33);
          v26 = v30;
        }
        v28 = 1;
        if ( v26 != 1 )
          v28 = v32;
        v29 = __OFADD__(v27, v9);
        v9 += v27;
        v8 = v39;
        v32 = v28;
        if ( v29 )
        {
          v9 = 0x8000000000000000LL;
          if ( v27 > 0 )
            v9 = 0x7FFFFFFFFFFFFFFFLL;
        }
        goto LABEL_9;
      }
    }
    else
    {
LABEL_23:
      sub_C8CC70((__int64)&v35, (__int64)v14, v17, v8, v16, a6);
      v8 = v39;
      if ( !v19 )
        goto LABEL_9;
      v20 = *(unsigned __int8 *)(v13 + 8);
      if ( v20 == 17 )
        goto LABEL_31;
    }
    if ( v20 != 18 )
      goto LABEL_9;
    ++v7;
    ++v11;
    v32 = 1;
  }
  while ( (_BYTE **)v10 != v7 );
LABEL_27:
  if ( !(_BYTE)v8 )
    _libc_free((unsigned __int64)v36);
  return v9;
}
