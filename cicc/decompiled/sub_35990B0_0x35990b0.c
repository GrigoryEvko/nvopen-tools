// Function: sub_35990B0
// Address: 0x35990b0
//
void __fastcall sub_35990B0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r15
  _DWORD *v8; // rdx
  __int64 v9; // r8
  int *v11; // rdx
  int v12; // eax
  int *v13; // rbx
  __int64 v14; // r11
  __int64 *v15; // r13
  __int64 *v16; // r15
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  _BYTE *v25; // rdx
  __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  char v28; // al
  unsigned __int64 v29; // rdx
  char v30; // si
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rcx
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rcx
  int v38; // [rsp+2Ch] [rbp-54h] BYREF
  _BYTE *v39; // [rsp+30h] [rbp-50h] BYREF
  __int64 v40; // [rsp+38h] [rbp-48h]
  _BYTE v41[64]; // [rsp+40h] [rbp-40h] BYREF

  if ( !a4 )
    return;
  v6 = *(_QWORD *)(a2 + 48);
  v7 = a2;
  v8 = (_DWORD *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return;
  v9 = a4;
  if ( (v6 & 7) != 0 )
  {
    if ( (v6 & 7) != 3 || !*v8 )
      return;
  }
  else
  {
    *(_QWORD *)(a2 + 48) = v8;
    v6 &= 0xFFFFFFFFFFFFFFF8LL;
  }
  v39 = v41;
  v40 = 0x200000000LL;
  v11 = (int *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_30;
  v12 = v6 & 7;
  if ( v12 )
  {
    if ( v12 != 3 )
      goto LABEL_30;
    v13 = v11 + 4;
    v14 = 2LL * *v11;
  }
  else
  {
    *(_QWORD *)(a2 + 48) = v11;
    v13 = (int *)(a2 + 48);
    v14 = 2;
  }
  v15 = (__int64 *)&v13[v14];
  if ( &v13[v14] == v13 )
  {
LABEL_30:
    v25 = v41;
    v26 = 0;
    goto LABEL_26;
  }
  v16 = (__int64 *)v13;
  do
  {
    v18 = *v16;
    if ( (*(_WORD *)(*v16 + 32) & 4) == 0 && (*(_BYTE *)(v18 + 37) & 0xF) == 0 && (*(_WORD *)(*v16 + 32) & 0x30) != 0x30 )
    {
      v19 = *(_QWORD *)v18;
      if ( *(_QWORD *)v18 )
      {
        if ( (v19 & 4) == 0 && (v19 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          if ( a4 == -1 || !(unsigned __int8)sub_3598FB0(a1, a3, &v38) )
          {
            v20 = *(_QWORD *)(a1 + 8);
            v21 = 0;
            v22 = 0;
LABEL_21:
            v18 = sub_2E7ACE0(v20, v18, v22, v21);
            goto LABEL_22;
          }
          v27 = *(_QWORD *)(v18 + 24);
          v20 = *(_QWORD *)(a1 + 8);
          v21 = 0;
          if ( (v27 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
          {
LABEL_32:
            v22 = v38 * a4;
            goto LABEL_21;
          }
          v28 = *(_BYTE *)(v18 + 24);
          v29 = v27 >> 3;
          v30 = v28 & 2;
          if ( (v28 & 6) == 2 || (v28 & 1) != 0 )
          {
            v33 = v29 >> 45;
            v34 = v29 >> 29;
            if ( !v30 )
              LODWORD(v33) = v34;
            v35 = (unsigned __int64)(((_DWORD)v33 + 7) & 0xFFFFFFF8) << 29;
          }
          else
          {
            v31 = v29 >> 29;
            if ( v30 )
              LODWORD(v31) = v29 >> 45;
            v32 = ((unsigned __int64)((unsigned int)v31 * (unsigned __int16)(v29 >> 5)) + 7) >> 3;
            if ( (v29 & 1) != 0 )
            {
              v21 = (v32 << 35) | 0x10C;
              goto LABEL_32;
            }
            v35 = (unsigned __int64)(unsigned int)(8 * v32) << 29;
          }
          v21 = 8 * v35 + 1;
          goto LABEL_32;
        }
      }
    }
LABEL_22:
    v23 = (unsigned int)v40;
    v24 = (unsigned int)v40 + 1LL;
    if ( v24 > HIDWORD(v40) )
    {
      sub_C8D5F0((__int64)&v39, v41, v24, 8u, v9, a6);
      v23 = (unsigned int)v40;
    }
    ++v16;
    *(_QWORD *)&v39[8 * v23] = v18;
    LODWORD(v40) = v40 + 1;
  }
  while ( v15 != v16 );
  v7 = a2;
  v25 = v39;
  v26 = (unsigned int)v40;
LABEL_26:
  sub_2E86A90(v7, *(_QWORD *)(a1 + 8), v25, v26);
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
}
