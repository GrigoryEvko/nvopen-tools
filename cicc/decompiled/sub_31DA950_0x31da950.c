// Function: sub_31DA950
// Address: 0x31da950
//
void __fastcall sub_31DA950(__int64 a1, __int64 a2)
{
  _BYTE *v3; // r12
  unsigned int v4; // r14d
  unsigned __int64 v5; // rax
  bool v6; // zf
  unsigned __int64 *v7; // r13
  __int64 v8; // r15
  __int64 v9; // r14
  unsigned __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // [rsp+8h] [rbp-68h]
  unsigned int v18; // [rsp+18h] [rbp-58h]
  unsigned int v19; // [rsp+1Ch] [rbp-54h]
  unsigned __int64 v20; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-48h]
  _QWORD v22[8]; // [rsp+30h] [rbp-40h] BYREF

  v3 = (_BYTE *)sub_31DA930(a2);
  v4 = *(_DWORD *)(a1 + 32);
  v21 = v4;
  if ( v4 > 0x40 )
  {
    v7 = &v20;
    sub_C43780((__int64)&v20, (const void **)(a1 + 24));
    v19 = v4 >> 6;
    v18 = v4 & 0x3F;
    if ( (v4 & 0x3F) != 0 )
    {
      v4 = v21;
      if ( *v3 )
      {
        v5 = v20;
        v18 = 8 * ((v18 - 1) >> 3) + 8;
        v15 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18);
        if ( v21 > 0x40 )
        {
          v17 = *(_QWORD *)v20 & v15;
          sub_C482E0((__int64)&v20, v18);
          if ( v21 > 0x40 )
            v7 = (unsigned __int64 *)v20;
          goto LABEL_8;
        }
        v17 = v20 & v15;
LABEL_33:
        v16 = v5 >> v18;
        if ( v18 == v4 )
          v16 = 0;
        v20 = v16;
LABEL_24:
        if ( v19 )
          goto LABEL_5;
        goto LABEL_18;
      }
      if ( v21 <= 0x40 )
        goto LABEL_4;
      v7 = (unsigned __int64 *)v20;
      v17 = *(_QWORD *)(v20 + 8LL * v19);
    }
    else
    {
      v17 = 0;
      if ( v21 <= 0x40 )
      {
LABEL_8:
        v8 = 1;
        v9 = v19 + 1;
        do
        {
          if ( *v3 )
            v10 = v7[v19 - (unsigned int)v8];
          else
            v10 = v7[v8 - 1];
          ++v8;
          (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64))(**(_QWORD **)(a2 + 224) + 536LL))(
            *(_QWORD *)(a2 + 224),
            v10,
            8);
        }
        while ( v9 != v8 );
LABEL_17:
        if ( !v18 )
          goto LABEL_19;
        goto LABEL_18;
      }
      v7 = (unsigned __int64 *)v20;
    }
    if ( !v19 )
      goto LABEL_17;
    goto LABEL_8;
  }
  v5 = *(_QWORD *)(a1 + 24);
  v20 = v5;
  v18 = v4 & 0x3F;
  if ( (v4 & 0x3F) == 0 )
  {
    v19 = v4 >> 6;
    if ( !(v4 >> 6) )
      return;
    v17 = 0;
    v7 = &v20;
    goto LABEL_8;
  }
  v6 = *v3 == 0;
  v19 = v4 >> 6;
  if ( !v6 )
  {
    v18 = 8 * ((v18 - 1) >> 3) + 8;
    v17 = v5 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18));
    if ( v4 <= 0x3F )
      goto LABEL_24;
    goto LABEL_33;
  }
LABEL_4:
  v17 = *(&v20 + v19);
  if ( v19 )
  {
LABEL_5:
    v7 = &v20;
    goto LABEL_8;
  }
LABEL_18:
  v11 = sub_31DA930(a2);
  v12 = sub_9208B0(v11, *(_QWORD *)(a1 + 8));
  v22[1] = v13;
  v22[0] = (unsigned __int64)(v12 + 7) >> 3;
  v14 = sub_CA1930(v22);
  (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64))(**(_QWORD **)(a2 + 224) + 536LL))(
    *(_QWORD *)(a2 + 224),
    v17,
    v14 - 8 * v19);
LABEL_19:
  if ( v21 > 0x40 )
  {
    if ( v20 )
      j_j___libc_free_0_0(v20);
  }
}
